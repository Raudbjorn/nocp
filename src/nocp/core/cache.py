"""
Cache Module: Tool Result Caching

Provides in-memory LRU caching and optional ChromaDB integration for distributed caching.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from ..models.contracts import ToolRequest, ToolResult

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[ToolResult]:
        """Get a value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in cache with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

    @abstractmethod
    async def get_async(self, key: str) -> Optional[ToolResult]:
        """Async version of get."""
        pass

    @abstractmethod
    async def set_async(self, key: str, value: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """Async version of set."""
        pass


class LRUCache(CacheBackend):
    """
    In-memory LRU (Least Recently Used) cache for tool results.

    Features:
    - Configurable max size
    - TTL (time-to-live) support
    - Thread-safe operations
    - Hit/miss statistics

    Example:
        cache = LRUCache(max_size=1000, default_ttl=3600)
        cache.set("key1", tool_result)
        result = cache.get("key1")
    """

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = 3600):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default time-to-live in seconds (None = no expiration)
        """
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _generate_key(self, request: ToolRequest) -> str:
        """
        Generate a cache key from a ToolRequest.

        Uses tool_id and parameters to create a deterministic hash.

        Args:
            request: ToolRequest to generate key for

        Returns:
            Cache key string
        """
        # Sort parameters for deterministic hashing
        params_str = json.dumps(request.parameters, sort_keys=True)
        key_data = f"{request.tool_id}:{params_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired."""
        if entry["expires_at"] is None:
            return False
        return datetime.now() > entry["expires_at"]

    def get(self, key: str) -> Optional[ToolResult]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached ToolResult or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if self._is_expired(entry):
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1

            logger.debug(f"Cache hit for key: {key[:16]}...")
            return entry["value"]

    def set(self, key: str, value: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """
        Set a value in cache with optional TTL.

        Args:
            key: Cache key
            value: ToolResult to cache
            ttl_seconds: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            # Determine expiration time
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires_at = None if ttl is None else datetime.now() + timedelta(seconds=ttl)

            # Add to cache
            self._cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now()
            }

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Evict oldest if over max size
            if len(self._cache) > self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
                logger.debug(f"Evicted oldest cache entry: {oldest_key[:16]}...")

            logger.debug(f"Cached result for key: {key[:16]}... (TTL: {ttl}s)")

    def delete(self, key: str) -> None:
        """Delete a value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Deleted cache entry: {key[:16]}...")

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {count} cache entries")

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hit rate, size, and other metrics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

            return {
                "backend": "in_memory_lru",
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "default_ttl": self._default_ttl
            }

    async def get_async(self, key: str) -> Optional[ToolResult]:
        """Async version of get (delegates to sync implementation)."""
        return self.get(key)

    async def set_async(self, key: str, value: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """Async version of set (delegates to sync implementation)."""
        self.set(key, value, ttl_seconds)

    def get_by_request(self, request: ToolRequest) -> Optional[ToolResult]:
        """
        Get cached result for a ToolRequest.

        Args:
            request: ToolRequest to look up

        Returns:
            Cached ToolResult or None
        """
        key = self._generate_key(request)
        return self.get(key)

    def set_by_request(self, request: ToolRequest, result: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """
        Cache a ToolResult for a ToolRequest.

        Args:
            request: ToolRequest that generated the result
            result: ToolResult to cache
            ttl_seconds: Time-to-live in seconds
        """
        key = self._generate_key(request)
        self.set(key, result, ttl_seconds)

    async def get_by_request_async(self, request: ToolRequest) -> Optional[ToolResult]:
        """Async version of get_by_request."""
        key = self._generate_key(request)
        return await self.get_async(key)

    async def set_by_request_async(
        self,
        request: ToolRequest,
        result: ToolResult,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Async version of set_by_request."""
        key = self._generate_key(request)
        await self.set_async(key, result, ttl_seconds)


class ChromaDBCache(CacheBackend):
    """
    ChromaDB-backed distributed cache for tool results.

    Features:
    - Distributed caching across multiple processes/servers
    - Automatic TTL support with metadata tracking
    - JSON serialization/deserialization
    - Async support with asyncio

    Example:
        cache = ChromaDBCache(persist_directory="./chroma_cache")
        cache.set("key1", tool_result)
        result = cache.get("key1")
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "nocp_cache",
        default_ttl: Optional[int] = 3600,
    ):
        """
        Initialize ChromaDB cache.

        Args:
            persist_directory: Directory to persist ChromaDB data (None = in-memory)
            collection_name: Name of the ChromaDB collection
            default_ttl: Default time-to-live in seconds
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB support requires 'chromadb' package. "
                "Install with: pip install chromadb"
            )

        # Create ChromaDB client
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Using persistent ChromaDB at {persist_directory}")
        else:
            self._client = chromadb.Client()
            logger.info("Using in-memory ChromaDB")

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "NOCP tool result cache"}
        )

        self._default_ttl = default_ttl
        self._collection_name = collection_name

        # Statistics
        self._hits = 0
        self._misses = 0

        logger.info(f"ChromaDB cache initialized with collection '{collection_name}'")

    def _serialize(self, result: ToolResult) -> Dict[str, Any]:
        """Serialize ToolResult to dictionary."""
        return result.model_dump(mode='json')

    def _deserialize(self, data: Dict[str, Any]) -> ToolResult:
        """Deserialize dictionary to ToolResult."""
        return ToolResult(**data)

    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired based on metadata."""
        if metadata.get("expires_at") is None:
            return False
        expires_at = float(metadata["expires_at"])
        return time.time() > expires_at

    def get(self, key: str) -> Optional[ToolResult]:
        """Get a value from ChromaDB cache."""
        try:
            result = self._collection.get(ids=[key], include=["metadatas", "documents"])

            if not result["ids"]:
                self._misses += 1
                logger.debug(f"Cache miss for key: {key[:16]}...")
                return None

            # Check expiration
            metadata = result["metadatas"][0]
            if self._is_expired(metadata):
                self.delete(key)
                self._misses += 1
                logger.debug(f"Cache expired for key: {key[:16]}...")
                return None

            # Deserialize from metadata
            cache_data = json.loads(metadata["tool_result"])
            self._hits += 1
            logger.debug(f"Cache hit for key: {key[:16]}...")
            return self._deserialize(cache_data)

        except Exception as e:
            logger.warning(f"Error getting cache key {key[:16]}: {e}")
            self._misses += 1
            return None

    def set(self, key: str, value: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in ChromaDB cache with optional TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = None if ttl is None else time.time() + ttl

        # Serialize ToolResult to JSON string for metadata storage
        serialized_data = self._serialize(value)

        metadata = {
            "tool_result": json.dumps(serialized_data),
            "created_at": str(time.time()),
            "expires_at": str(expires_at) if expires_at else "null",
            "ttl": str(ttl) if ttl else "null"
        }

        try:
            # Check if key exists
            existing = self._collection.get(ids=[key])

            if existing["ids"]:
                # Update existing entry
                self._collection.update(
                    ids=[key],
                    metadatas=[metadata],
                    documents=[f"cache_entry_{key[:16]}"]
                )
            else:
                # Add new entry
                self._collection.add(
                    ids=[key],
                    metadatas=[metadata],
                    documents=[f"cache_entry_{key[:16]}"]
                )

            logger.debug(f"Cached result in ChromaDB for key: {key[:16]}... (TTL: {ttl}s)")

        except Exception as e:
            logger.error(f"Error setting cache key {key[:16]}: {e}")

    def delete(self, key: str) -> None:
        """Delete a value from ChromaDB cache."""
        try:
            self._collection.delete(ids=[key])
            logger.debug(f"Deleted ChromaDB cache entry: {key[:16]}...")
        except Exception as e:
            logger.warning(f"Error deleting cache key {key[:16]}: {e}")

    def clear(self) -> None:
        """Clear all cached values by deleting and recreating the collection."""
        try:
            # Get count before deletion
            count = self._collection.count()

            # Delete the collection
            self._client.delete_collection(name=self._collection_name)

            # Recreate the collection
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"description": "NOCP tool result cache"}
            )

            logger.info(f"Cleared {count} ChromaDB cache entries")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def stats(self) -> Dict[str, Any]:
        """Get ChromaDB cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

        return {
            "backend": "chromadb",
            "size": self._collection.count(),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "default_ttl": self._default_ttl,
            "collection_name": self._collection_name
        }

    async def get_async(self, key: str) -> Optional[ToolResult]:
        """Async version of get (delegates to sync with asyncio.to_thread)."""
        return await asyncio.to_thread(self.get, key)

    async def set_async(self, key: str, value: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """Async version of set (delegates to sync with asyncio.to_thread)."""
        await asyncio.to_thread(self.set, key, value, ttl_seconds)

    def _generate_key(self, request: ToolRequest) -> str:
        """Generate a cache key from a ToolRequest."""
        params_str = json.dumps(request.parameters, sort_keys=True)
        key_data = f"{request.tool_id}:{params_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get_by_request(self, request: ToolRequest) -> Optional[ToolResult]:
        """Get cached result for a ToolRequest."""
        key = self._generate_key(request)
        return self.get(key)

    def set_by_request(self, request: ToolRequest, result: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """Cache a ToolResult for a ToolRequest."""
        key = self._generate_key(request)
        self.set(key, result, ttl_seconds)

    async def get_by_request_async(self, request: ToolRequest) -> Optional[ToolResult]:
        """Async version of get_by_request."""
        key = self._generate_key(request)
        return await self.get_async(key)

    async def set_by_request_async(
        self,
        request: ToolRequest,
        result: ToolResult,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Async version of set_by_request."""
        key = self._generate_key(request)
        await self.set_async(key, result, ttl_seconds)


class CacheConfig:
    """Configuration for cache backend selection."""

    def __init__(
        self,
        backend: str = "memory",
        max_size: int = 1000,
        default_ttl: Optional[int] = 3600,
        chromadb_persist_dir: Optional[str] = None,
        chromadb_collection_name: str = "nocp_cache",
        enabled: bool = True
    ):
        """
        Initialize cache configuration.

        Args:
            backend: Cache backend ("memory" or "chromadb")
            max_size: Max size for in-memory cache
            default_ttl: Default TTL in seconds (None = no expiration)
            chromadb_persist_dir: ChromaDB persistence directory (None = in-memory)
            chromadb_collection_name: ChromaDB collection name
            enabled: Enable/disable caching
        """
        self.backend = backend
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.chromadb_persist_dir = chromadb_persist_dir
        self.chromadb_collection_name = chromadb_collection_name
        self.enabled = enabled

    def create_backend(self) -> Optional[CacheBackend]:
        """
        Create a cache backend based on configuration.

        Returns:
            CacheBackend instance or None if caching is disabled
        """
        if not self.enabled:
            logger.info("Caching is disabled")
            return None

        if self.backend == "memory":
            logger.info(f"Using in-memory LRU cache (max_size={self.max_size}, ttl={self.default_ttl}s)")
            return LRUCache(max_size=self.max_size, default_ttl=self.default_ttl)
        elif self.backend == "chromadb":
            logger.info(f"Using ChromaDB cache at {self.chromadb_persist_dir or 'in-memory'}")
            return ChromaDBCache(
                persist_directory=self.chromadb_persist_dir,
                collection_name=self.chromadb_collection_name,
                default_ttl=self.default_ttl
            )
        else:
            raise ValueError(f"Unknown cache backend: {self.backend}")
