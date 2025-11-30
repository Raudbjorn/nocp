"""
Tests for conversation history compaction and persistence.

Tests cover:
- Multi-turn conversation scenarios
- Roll-up summarization
- State persistence across sessions
- History compaction triggers
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nocp.core.persistence import PersistenceManager
from nocp.models.context import (
    ContextSnapshot,
    ConversationMessage,
    PersistentContext,
    TransientContext,
)
from nocp.modules.context_manager import ContextManager


class TestConversationHistoryCompaction:
    """Tests for conversation history compaction."""

    def setup_method(self):
        """Set up test fixtures."""
        # Set dummy API key for testing
        os.environ["GEMINI_API_KEY"] = "test_key_for_testing"

        # Mock the student model to avoid actual API calls
        with patch("nocp.modules.context_manager.genai.GenerativeModel") as mock_model:
            mock_instance = Mock()
            mock_model.return_value = mock_instance
            self.context_manager = ContextManager()
            self.context_manager.student_model = mock_instance

    def teardown_method(self):
        """Clean up after tests."""
        # Reset config
        from nocp.core.config import reset_config

        reset_config()

    def test_no_compaction_needed_below_threshold(self):
        """Test that compaction is skipped when below threshold."""
        # Create transient context with few messages
        transient_ctx = TransientContext(
            current_query="What is the weather?",
            conversation_history=[
                ConversationMessage(role="user", content="Hello", token_count=2),
                ConversationMessage(role="assistant", content="Hi there!", token_count=3),
            ],
            max_history_tokens=50000,
        )

        result = self.context_manager.compact_conversation_history(transient_ctx)

        assert result is None
        assert len(transient_ctx.conversation_history) == 2

    def test_compaction_with_many_messages(self):
        """Test that compaction occurs with many messages exceeding threshold."""
        # Create many messages exceeding threshold
        messages = []
        for i in range(20):
            messages.append(
                ConversationMessage(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i} with some content " * 100,  # ~500 tokens each
                    token_count=500,
                )
            )

        transient_ctx = TransientContext(
            current_query="Continue conversation",
            conversation_history=messages,
            max_history_tokens=5000,  # 20 messages * 500 tokens = 10,000 tokens > threshold
        )

        persistent_ctx = PersistentContext(session_id="test_session")

        result = self.context_manager.compact_conversation_history(
            transient_ctx,
            persistent_ctx=persistent_ctx,
            keep_recent=5,
        )

        # Should have performed compaction
        assert result is not None
        assert result.compression_method == "history_compaction"
        assert result.net_savings > 0

        # Should keep only recent messages + 1 summary
        assert len(transient_ctx.conversation_history) == 6  # 1 summary + 5 recent

        # First message should be summary
        assert transient_ctx.conversation_history[0].role == "system"
        assert "[Conversation Summary]" in transient_ctx.conversation_history[0].content

        # Persistent context should be updated
        assert persistent_ctx.conversation_summary is not None
        assert persistent_ctx.summary_generations == 1

    def test_rollup_summarization(self):
        """Test roll-up summarization with existing summary."""
        # Create initial summary in persistent context
        persistent_ctx = PersistentContext(
            session_id="test_session",
            conversation_summary="User discussed project requirements and decided to use Python.",
            summary_generations=1,
        )

        # Create new messages
        messages = []
        for i in range(15):
            messages.append(
                ConversationMessage(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Additional discussion {i}: Details about implementation and architecture."
                    * 50,
                    token_count=400,
                )
            )

        transient_ctx = TransientContext(
            current_query="Continue",
            conversation_history=messages,
            max_history_tokens=3000,  # 15 * 400 = 6000 > threshold
            turn_number=20,
        )

        result = self.context_manager.compact_conversation_history(
            transient_ctx,
            persistent_ctx=persistent_ctx,
            keep_recent=5,
        )

        # Should have performed roll-up compaction
        assert result is not None

        # Summary generations should increment
        assert persistent_ctx.summary_generations == 2

        # Last compaction turn should be updated
        assert persistent_ctx.last_compaction_turn == 20

        # Summary should be updated (not same as original)
        assert (
            persistent_ctx.conversation_summary
            != "User discussed project requirements and decided to use Python."
        )

    def test_compression_metrics_updated(self):
        """Test that compression metrics are tracked in persistent context."""
        persistent_ctx = PersistentContext(session_id="test_session")

        messages = [
            ConversationMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}" * 200,
                token_count=400,
            )
            for i in range(12)
        ]

        transient_ctx = TransientContext(
            current_query="Test",
            conversation_history=messages,
            max_history_tokens=3000,
        )

        result = self.context_manager.compact_conversation_history(
            transient_ctx,
            persistent_ctx=persistent_ctx,
        )

        assert result is not None

        # Compression metrics should be updated
        assert persistent_ctx.total_compressions == 1
        assert persistent_ctx.total_compression_savings > 0


class TestPersistenceManager:
    """Tests for conversation state persistence."""

    def setup_method(self):
        """Set up test fixtures with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_manager = PersistenceManager(storage_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up temporary storage."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_save_session(self):
        """Test creating and saving a new session."""
        session_id = "test_session_001"

        context = self.persistence_manager.create_session(
            session_id=session_id,
            system_instructions="Custom system instructions",
        )

        assert context.session_id == session_id
        assert context.system_instructions == "Custom system instructions"
        assert context.state == "active"

        # Verify file was created
        session_file = Path(self.temp_dir) / f"{session_id}.json"
        assert session_file.exists()

    def test_load_existing_session(self):
        """Test loading an existing session from disk."""
        session_id = "test_session_002"

        # Create and save
        original_context = self.persistence_manager.create_session(session_id)
        original_context.total_turns = 10
        original_context.total_tokens_processed = 5000
        original_context.conversation_summary = "Test summary"
        self.persistence_manager.save_persistent_context(original_context)

        # Clear cache
        self.persistence_manager._session_cache.clear()

        # Load from disk
        loaded_context = self.persistence_manager.load_persistent_context(session_id)

        assert loaded_context is not None
        assert loaded_context.session_id == session_id
        assert loaded_context.total_turns == 10
        assert loaded_context.total_tokens_processed == 5000
        assert loaded_context.conversation_summary == "Test summary"

    def test_get_or_create_session(self):
        """Test get_or_create behavior."""
        session_id = "test_session_003"

        # First call should create
        context1 = self.persistence_manager.get_or_create_session(session_id)
        assert context1.session_id == session_id
        assert context1.total_turns == 0

        # Modify and save
        context1.total_turns = 5
        self.persistence_manager.save_persistent_context(context1)

        # Clear cache
        self.persistence_manager._session_cache.clear()

        # Second call should load existing
        context2 = self.persistence_manager.get_or_create_session(session_id)
        assert context2.total_turns == 5

    def test_archive_session(self):
        """Test archiving a session."""
        session_id = "test_session_004"

        context = self.persistence_manager.create_session(session_id)
        assert context.state == "active"

        # Archive
        success = self.persistence_manager.archive_session(session_id)
        assert success

        # Load and verify archived
        archived_context = self.persistence_manager.load_persistent_context(session_id)
        assert archived_context.state == "archived"

    def test_delete_session(self):
        """Test deleting a session."""
        session_id = "test_session_005"

        self.persistence_manager.create_session(session_id)

        # Verify exists
        session_file = Path(self.temp_dir) / f"{session_id}.json"
        assert session_file.exists()

        # Delete
        success = self.persistence_manager.delete_session(session_id)
        assert success

        # Verify deleted
        assert not session_file.exists()
        loaded = self.persistence_manager.load_persistent_context(session_id)
        assert loaded is None

    def test_list_sessions(self):
        """Test listing all sessions."""
        # Create multiple sessions
        self.persistence_manager.create_session("session_a")
        self.persistence_manager.create_session("session_b")
        self.persistence_manager.create_session("session_c")

        # Archive one
        self.persistence_manager.archive_session("session_b")

        # List active sessions
        sessions = self.persistence_manager.list_sessions(include_archived=False)
        assert "session_a" in sessions
        assert "session_c" in sessions
        assert "session_b" not in sessions

        # List all sessions
        all_sessions = self.persistence_manager.list_sessions(include_archived=True)
        assert "session_a" in all_sessions
        assert "session_b" in all_sessions
        assert "session_c" in all_sessions

    def test_save_snapshot(self):
        """Test saving context snapshots."""
        session_id = "test_session_006"

        transient = TransientContext(
            current_query="Test query",
            turn_number=5,
        )

        persistent = self.persistence_manager.create_session(session_id)

        snapshot = ContextSnapshot(
            transient=transient,
            persistent=persistent,
            total_context_tokens=1000,
        )

        success = self.persistence_manager.save_snapshot(snapshot, session_id)
        assert success

        # Verify snapshot file exists
        snapshots_dir = Path(self.temp_dir) / "snapshots" / session_id
        assert snapshots_dir.exists()
        snapshot_files = list(snapshots_dir.glob("snapshot_*.json"))
        assert len(snapshot_files) == 1


class TestMultiTurnConversationIntegration:
    """Integration tests for multi-turn conversations with persistence."""

    def setup_method(self):
        """Set up test fixtures."""
        # Set dummy API key for testing
        os.environ["GEMINI_API_KEY"] = "test_key_for_testing"

        self.temp_dir = tempfile.mkdtemp()
        self.persistence_manager = PersistenceManager(storage_dir=self.temp_dir)

        # Mock the student model to avoid actual API calls
        with patch("nocp.modules.context_manager.genai.GenerativeModel") as mock_model:
            mock_instance = Mock()
            mock_model.return_value = mock_instance
            self.context_manager = ContextManager()
            self.context_manager.student_model = mock_instance

    def teardown_method(self):
        """Clean up temporary storage."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Reset config
        from nocp.core.config import reset_config

        reset_config()

    def test_multi_turn_conversation_with_compaction(self):
        """Test a multi-turn conversation that triggers compaction."""
        session_id = "multi_turn_session"
        persistent_ctx = self.persistence_manager.create_session(session_id)

        # Simulate 30 turns of conversation
        messages = []
        for turn in range(1, 31):
            # User message
            messages.append(
                ConversationMessage(
                    role="user",
                    content=f"Turn {turn}: User asks a detailed question about the system architecture and implementation details."
                    * 30,
                    token_count=300,
                )
            )

            # Assistant message
            messages.append(
                ConversationMessage(
                    role="assistant",
                    content=f"Turn {turn}: Assistant provides comprehensive answer with code examples and explanations."
                    * 40,
                    token_count=400,
                )
            )

            # Check if compaction needed every 5 turns
            if turn % 5 == 0:
                transient_ctx = TransientContext(
                    current_query=f"Turn {turn} query",
                    conversation_history=messages.copy(),
                    max_history_tokens=8000,  # Should trigger compaction around turn 15
                    turn_number=turn,
                )

                compression_result = self.context_manager.compact_conversation_history(
                    transient_ctx,
                    persistent_ctx=persistent_ctx,
                    keep_recent=5,
                )

                if compression_result:
                    # Update messages to compacted version
                    messages = transient_ctx.conversation_history.copy()

                    # Save persistent context
                    self.persistence_manager.save_persistent_context(persistent_ctx)

        # By turn 30, should have had multiple compressions
        assert persistent_ctx.summary_generations >= 1
        assert persistent_ctx.conversation_summary is not None
        assert persistent_ctx.total_compressions >= 1

        # Final message list should be much smaller than 60 messages (30 turns * 2)
        assert len(messages) < 60

    def test_conversation_state_recovery(self):
        """Test recovering conversation state across sessions."""
        session_id = "recovery_session"

        # Session 1: Create conversation and compact
        persistent_ctx_1 = self.persistence_manager.create_session(session_id)

        messages = [
            ConversationMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}: Detailed conversation content." * 50,
                token_count=400,
            )
            for i in range(20)
        ]

        transient_ctx_1 = TransientContext(
            current_query="Session 1 query",
            conversation_history=messages,
            max_history_tokens=5000,
            turn_number=10,
        )

        self.context_manager.compact_conversation_history(
            transient_ctx_1,
            persistent_ctx=persistent_ctx_1,
        )

        # Save state
        self.persistence_manager.save_persistent_context(persistent_ctx_1)

        summary_gen_1 = persistent_ctx_1.summary_generations
        summary_content_1 = persistent_ctx_1.conversation_summary

        # Clear cache (simulate restart)
        self.persistence_manager._session_cache.clear()

        # Session 2: Load state and continue
        persistent_ctx_2 = self.persistence_manager.load_persistent_context(session_id)

        assert persistent_ctx_2 is not None
        assert persistent_ctx_2.summary_generations == summary_gen_1
        assert persistent_ctx_2.conversation_summary == summary_content_1

        # Add more messages and compact again
        new_messages = [
            ConversationMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"New message {i}: More conversation." * 50,
                token_count=400,
            )
            for i in range(15)
        ]

        transient_ctx_2 = TransientContext(
            current_query="Session 2 query",
            conversation_history=new_messages,
            max_history_tokens=4000,
            turn_number=20,
        )

        self.context_manager.compact_conversation_history(
            transient_ctx_2,
            persistent_ctx=persistent_ctx_2,
        )

        # Summary should be updated (roll-up)
        assert persistent_ctx_2.summary_generations > summary_gen_1
        assert persistent_ctx_2.conversation_summary != summary_content_1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
