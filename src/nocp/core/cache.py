"""
Cache Module: Tool Result Caching

Provides in-memory LRU caching and optional Redis integration for distributed caching.
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


class RedisCache(CacheBackend):
    """
    Redis-backed distributed cache for tool results.

    Features:
    - Distributed caching across multiple processes/servers
    - Automatic TTL support
    - JSON serialization/deserialization
    - Async support with aioredis

    Example:
        cache = RedisCache(host="localhost", port=6379)
        cache.set("key1", tool_result)
        result = cache.get("key1")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: Optional[int] = 3600,
        key_prefix: str = "nocp:cache:"
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
            default_ttl: Default time-to-live in seconds
            key_prefix: Prefix for all cache keys
        """
        try:
            import redis
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "Redis support requires 'redis' package. "
                "Install with: pip install redis"
            )

        self._redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # We'll handle serialization
        )

        self._aioredis = aioredis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False
        )

        self._default_ttl = default_ttl
        self._key_prefix = key_prefix

        # Test connection
        try:
            self._redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _make_key(self, key: str) -> str:
        """Add prefix to cache key."""
        return f"{self._key_prefix}{key}"

    def _serialize(self, result: ToolResult) -> bytes:
        """Serialize ToolResult to bytes."""
        data = result.model_dump(mode='json')
        return json.dumps(data).encode('utf-8')

    def _deserialize(self, data: bytes) -> ToolResult:
        """Deserialize bytes to ToolResult."""
        obj = json.loads(data.decode('utf-8'))
        return ToolResult(**obj)

    def get(self, key: str) -> Optional[ToolResult]:
        """Get a value from Redis cache."""
        redis_key = self._make_key(key)
        data = self._redis.get(redis_key)

        if data is None:
            logger.debug(f"Cache miss for key: {key[:16]}...")
            return None

        logger.debug(f"Cache hit for key: {key[:16]}...")
        return self._deserialize(data)

    def set(self, key: str, value: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in Redis cache with optional TTL."""
        redis_key = self._make_key(key)
        data = self._serialize(value)
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        if ttl is not None:
            self._redis.setex(redis_key, ttl, data)
        else:
            self._redis.set(redis_key, data)

        logger.debug(f"Cached result in Redis for key: {key[:16]}... (TTL: {ttl}s)")

    def delete(self, key: str) -> None:
        """Delete a value from Redis cache."""
        redis_key = self._make_key(key)
        self._redis.delete(redis_key)
        logger.debug(f"Deleted Redis cache entry: {key[:16]}...")

    def clear(self) -> None:
        """Clear all cached values with the key prefix."""
        pattern = f"{self._key_prefix}*"
        # Use SCAN to avoid blocking the server, as KEYS can be slow on large databases
        keys = list(self._redis.scan_iter(pattern))
        if keys:
            self._redis.delete(*keys)
            logger.info(f"Cleared {len(keys)} Redis cache entries")

    def stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        info = self._redis.info("stats")
        pattern = f"{self._key_prefix}*"
        key_count = len(self._redis.keys(pattern))

        return {
            "backend": "redis",
            "size": key_count,
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "default_ttl": self._default_ttl,
            "connected": True
        }

    async def get_async(self, key: str) -> Optional[ToolResult]:
        """Async version of get using aioredis."""
        redis_key = self._make_key(key)
        data = await self._aioredis.get(redis_key)

        if data is None:
            logger.debug(f"Cache miss for key: {key[:16]}...")
            return None

        logger.debug(f"Cache hit for key: {key[:16]}...")
        return self._deserialize(data)

    async def set_async(self, key: str, value: ToolResult, ttl_seconds: Optional[int] = None) -> None:
        """Async version of set using aioredis."""
        redis_key = self._make_key(key)
        data = self._serialize(value)
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        if ttl is not None:
            await self._aioredis.setex(redis_key, ttl, data)
        else:
            await self._aioredis.set(redis_key, data)

        logger.debug(f"Cached result in Redis for key: {key[:16]}... (TTL: {ttl}s)")

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
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize cache configuration.

        Args:
            backend: Cache backend ("memory" or "redis")
            max_size: Max size for in-memory cache
            default_ttl: Default TTL in seconds (None = no expiration)
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            redis_password: Redis password
            enabled: Enable/disable caching
        """
        self.backend = backend
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
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
        elif self.backend == "redis":
            logger.info(f"Using Redis cache at {self.redis_host}:{self.redis_port}")
            return RedisCache(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                default_ttl=self.default_ttl
            )
        else:
            raise ValueError(f"Unknown cache backend: {self.backend}")
