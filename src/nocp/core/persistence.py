"""
Persistence manager for conversation state across sessions.

Handles:
- Saving/loading persistent context to disk
- Session management
- State recovery
"""

import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from ..models.context import PersistentContext, TransientContext, ContextSnapshot
from ..utils.logging import get_logger


class PersistenceManager:
    """
    Manages persistent storage of conversation state.

    Responsibilities:
    - Save/load PersistentContext to/from disk
    - Handle session management
    - Provide state recovery mechanisms
    - Maintain conversation history across restarts
    """

    def __init__(self, storage_dir: str = "./data/sessions"):
        """
        Initialize the persistence manager.

        Args:
            storage_dir: Directory to store session data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

        # In-memory cache of active sessions
        self._session_cache: Dict[str, PersistentContext] = {}

    def save_persistent_context(
        self,
        context: PersistentContext,
        force: bool = False
    ) -> bool:
        """
        Save persistent context to disk.

        Args:
            context: Context to save
            force: Force save even if not dirty

        Returns:
            True if saved successfully
        """
        try:
            session_file = self._get_session_file(context.session_id)

            # Update cache
            self._session_cache[context.session_id] = context

            # Serialize to JSON
            context_dict = context.model_dump(mode='json')

            # Write atomically using temp file
            temp_file = session_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(context_dict, f, indent=2, default=str)

            # Atomic rename
            temp_file.replace(session_file)

            self.logger.info(
                "persistent_context_saved",
                session_id=context.session_id,
                total_turns=context.total_turns,
                total_tokens=context.total_tokens_processed,
            )

            return True

        except (IOError, OSError, TypeError, ValueError) as e:
            self.logger.error(
                "failed_to_save_persistent_context",
                session_id=context.session_id,
                error=str(e),
            )
            return False

    def load_persistent_context(
        self,
        session_id: str
    ) -> Optional[PersistentContext]:
        """
        Load persistent context from disk.

        Args:
            session_id: Session ID to load

        Returns:
            PersistentContext if found, None otherwise
        """
        # Check cache first
        if session_id in self._session_cache:
            self.logger.debug("persistent_context_cache_hit", session_id=session_id)
            return self._session_cache[session_id]

        try:
            session_file = self._get_session_file(session_id)

            if not session_file.exists():
                self.logger.info(
                    "persistent_context_not_found",
                    session_id=session_id,
                )
                return None

            with open(session_file, 'r') as f:
                context_dict = json.load(f)

            # Deserialize from JSON
            context = PersistentContext(**context_dict)

            # Update cache
            self._session_cache[session_id] = context

            self.logger.info(
                "persistent_context_loaded",
                session_id=session_id,
                total_turns=context.total_turns,
                created_at=context.created_at,
            )

            return context

        except (IOError, OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            self.logger.error(
                "failed_to_load_persistent_context",
                session_id=session_id,
                error=str(e),
            )
            return None

    def create_session(
        self,
        session_id: str,
        system_instructions: Optional[str] = None,
    ) -> PersistentContext:
        """
        Create a new persistent context session.

        Args:
            session_id: Unique session identifier
            system_instructions: Optional custom system instructions

        Returns:
            New PersistentContext instance
        """
        context = PersistentContext(
            session_id=session_id,
            system_instructions=system_instructions or PersistentContext.model_fields['system_instructions'].default,
        )

        # Save to disk
        self.save_persistent_context(context)

        self.logger.info("session_created", session_id=session_id)

        return context

    def get_or_create_session(
        self,
        session_id: str,
        system_instructions: Optional[str] = None,
    ) -> PersistentContext:
        """
        Get existing session or create new one.

        Args:
            session_id: Session identifier
            system_instructions: System instructions for new session

        Returns:
            PersistentContext instance
        """
        context = self.load_persistent_context(session_id)

        if context is None:
            context = self.create_session(session_id, system_instructions)

        return context

    def archive_session(self, session_id: str) -> bool:
        """
        Archive a session (mark as inactive).

        Args:
            session_id: Session to archive

        Returns:
            True if archived successfully
        """
        context = self.load_persistent_context(session_id)

        if context is None:
            return False

        context.state = "archived"
        return self.save_persistent_context(context, force=True)

    def delete_session(self, session_id: str) -> bool:
        """
        Permanently delete a session.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted successfully
        """
        try:
            session_file = self._get_session_file(session_id)

            if session_file.exists():
                session_file.unlink()

            # Remove from cache
            self._session_cache.pop(session_id, None)

            self.logger.info("session_deleted", session_id=session_id)
            return True

        except (IOError, OSError) as e:
            self.logger.error(
                "failed_to_delete_session",
                session_id=session_id,
                error=str(e),
            )
            return False

    def list_sessions(self, include_archived: bool = False) -> list[str]:
        """
        List all session IDs.

        Args:
            include_archived: Include archived sessions

        Returns:
            List of session IDs
        """
        try:
            session_files = self.storage_dir.glob("*.json")
            sessions = []

            for session_file in session_files:
                session_id = session_file.stem

                if not include_archived:
                    context = self.load_persistent_context(session_id)
                    if context and context.state == "archived":
                        continue

                sessions.append(session_id)

            return sorted(sessions)

        except (IOError, OSError) as e:
            self.logger.error("failed_to_list_sessions", error=str(e))
            return []

    def save_snapshot(
        self,
        snapshot: ContextSnapshot,
        session_id: str,
    ) -> bool:
        """
        Save a complete context snapshot for debugging/analysis.

        Args:
            snapshot: Context snapshot to save
            session_id: Session identifier

        Returns:
            True if saved successfully
        """
        try:
            snapshots_dir = self.storage_dir / "snapshots" / session_id
            snapshots_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            snapshot_file = snapshots_dir / f"snapshot_{timestamp}.json"

            snapshot_dict = snapshot.model_dump(mode='json')

            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_dict, f, indent=2, default=str)

            self.logger.info(
                "snapshot_saved",
                session_id=session_id,
                snapshot_file=str(snapshot_file),
            )

            return True

        except (IOError, OSError, TypeError, ValueError) as e:
            self.logger.error(
                "failed_to_save_snapshot",
                session_id=session_id,
                error=str(e),
            )
            return False

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        # Sanitize session_id to prevent directory traversal
        safe_session_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.storage_dir / f"{safe_session_id}.json"


# Global instance
_persistence_manager: Optional[PersistenceManager] = None


def get_persistence_manager(storage_dir: Optional[str] = None) -> PersistenceManager:
    """
    Get or create the global persistence manager instance.

    Args:
        storage_dir: Optional storage directory (only used on first call)

    Returns:
        PersistenceManager instance
    """
    global _persistence_manager

    if _persistence_manager is None:
        _persistence_manager = PersistenceManager(storage_dir or "./data/sessions")

    return _persistence_manager
