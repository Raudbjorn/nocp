"""
Advanced Example: API Aggregation with NOCP

This example demonstrates how to efficiently aggregate data from multiple APIs:
- Concurrent API calls for better performance
- Intelligent caching to reduce redundant requests
- Context optimization for large responses
- Error handling and retry logic
"""

import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nocp.core.act import ToolExecutor
from nocp.core.articulate import OutputSerializer
from nocp.core.assess import ContextManager
from nocp.core.async_modules import ConcurrentToolExecutor
from nocp.core.cache import LRUCache
from nocp.models.contracts import (
    ContextData,
    RetryConfig,
    SerializationRequest,
    ToolRequest,
    ToolResult,
    ToolType,
)
from pydantic import BaseModel, Field

# ============================================================================
# Data Models
# ============================================================================


class UserProfile(BaseModel):
    """User profile from API."""

    user_id: str
    name: str
    email: str
    bio: str = ""


class UserActivity(BaseModel):
    """User activity data."""

    user_id: str
    posts: int
    comments: int
    likes: int
    last_active: str


class UserMetrics(BaseModel):
    """User metrics from analytics."""

    user_id: str
    page_views: int
    session_duration: float
    bounce_rate: float


class AggregatedUserData(BaseModel):
    """Complete aggregated user data."""

    profile: UserProfile
    activity: UserActivity
    metrics: UserMetrics
    aggregated_at: datetime = Field(default_factory=datetime.now)


class DashboardData(BaseModel):
    """Dashboard with multiple users."""

    users: list[AggregatedUserData]
    total_users: int
    aggregation_time_ms: float
    cache_hits: int
    api_calls: int


# ============================================================================
# Simulated APIs
# ============================================================================


class MockAPIs:
    """Simulated external APIs for demonstration."""

    @staticmethod
    async def fetch_user_profile(user_id: str) -> dict:
        """Simulate fetching user profile from API."""
        await asyncio.sleep(0.1)  # Simulate network latency

        return {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
            "bio": f"Bio for user {user_id}",
        }

    @staticmethod
    async def fetch_user_activity(user_id: str) -> dict:
        """Simulate fetching user activity from API."""
        await asyncio.sleep(0.08)  # Simulate network latency

        import random

        return {
            "user_id": user_id,
            "posts": random.randint(0, 100),
            "comments": random.randint(0, 500),
            "likes": random.randint(0, 1000),
            "last_active": datetime.now().isoformat(),
        }

    @staticmethod
    async def fetch_user_metrics(user_id: str) -> dict:
        """Simulate fetching user metrics from analytics API."""
        await asyncio.sleep(0.12)  # Simulate network latency

        import random

        return {
            "user_id": user_id,
            "page_views": random.randint(100, 10000),
            "session_duration": round(random.uniform(1.0, 30.0), 2),
            "bounce_rate": round(random.uniform(0.1, 0.9), 2),
        }


# ============================================================================
# API Aggregation System
# ============================================================================


class APIAggregator:
    """Production-ready API aggregation system with NOCP optimizations."""

    def __init__(self):
        # Setup caching (critical for API aggregation to reduce costs)
        self.cache = LRUCache(max_size=10000, default_ttl=600)  # 10 min TTL

        # Setup executor with cache
        self.executor = ToolExecutor(cache=self.cache)

        # Setup concurrent executor (enables parallel API calls)
        self.concurrent = ConcurrentToolExecutor(
            self.executor, max_concurrent=10  # 10 concurrent API calls
        )

        # Context manager for large responses
        self.context_manager = ContextManager(
            compression_threshold=5000, target_compression_ratio=0.40, enable_litellm=False
        )

        # Output serializer
        self.serializer = OutputSerializer()

        # Statistics tracking
        self.stats = {"total_requests": 0, "cache_hits": 0, "api_calls": 0}

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register API tools."""

        @self.executor.register_async_tool("fetch_profile")
        async def fetch_profile(user_id: str) -> dict:
            """Fetch user profile from API."""
            self.stats["total_requests"] += 1
            self.stats["api_calls"] += 1
            # ToolExecutor handles caching automatically
            result = await MockAPIs.fetch_user_profile(user_id)
            return result

        @self.executor.register_async_tool("fetch_activity")
        async def fetch_activity(user_id: str) -> dict:
            """Fetch user activity from API."""
            self.stats["total_requests"] += 1
            self.stats["api_calls"] += 1
            # ToolExecutor handles caching automatically
            result = await MockAPIs.fetch_user_activity(user_id)
            return result

        @self.executor.register_async_tool("fetch_metrics")
        async def fetch_metrics(user_id: str) -> dict:
            """Fetch user metrics from analytics API."""
            self.stats["total_requests"] += 1
            self.stats["api_calls"] += 1
            # ToolExecutor handles caching automatically
            result = await MockAPIs.fetch_user_metrics(user_id)
            return result

    async def fetch_user_data(self, user_id: str) -> AggregatedUserData:
        """
        Fetch all data for a single user from multiple APIs.

        Uses concurrent execution to fetch from 3 APIs in parallel.
        """
        # Create requests for all 3 APIs
        requests = [
            ToolRequest(
                tool_id="fetch_profile",
                tool_type=ToolType.API_CALL,
                function_name="fetch_profile",
                parameters={"user_id": user_id},
                timeout_seconds=5,
                retry_config=RetryConfig(max_attempts=3),
            ),
            ToolRequest(
                tool_id="fetch_activity",
                tool_type=ToolType.API_CALL,
                function_name="fetch_activity",
                parameters={"user_id": user_id},
                timeout_seconds=5,
                retry_config=RetryConfig(max_attempts=3),
            ),
            ToolRequest(
                tool_id="fetch_metrics",
                tool_type=ToolType.API_CALL,
                function_name="fetch_metrics",
                parameters={"user_id": user_id},
                timeout_seconds=5,
                retry_config=RetryConfig(max_attempts=3),
            ),
        ]

        # Execute all requests concurrently
        results = await self.concurrent.execute_many_ordered(requests)

        # Build aggregated data
        profile = UserProfile(**results[0].data)
        activity = UserActivity(**results[1].data)
        metrics = UserMetrics(**results[2].data)

        return AggregatedUserData(profile=profile, activity=activity, metrics=metrics)

    async def fetch_dashboard(self, user_ids: list[str]) -> DashboardData:
        """
        Fetch dashboard data for multiple users.

        Demonstrates:
        - Concurrent API calls for multiple users
        - Caching across users
        - Context optimization for large datasets
        - Efficient serialization
        """
        print(f"\n{'='*60}")
        print(f"Fetching dashboard for {len(user_ids)} users")
        print("=" * 60)

        start_time = time.time()

        # Step 1: Fetch data for all users concurrently
        print("\n[1] Fetching user data concurrently...")
        fetch_start = time.time()

        tasks = [self.fetch_user_data(user_id) for user_id in user_ids]
        users_data = await asyncio.gather(*tasks)

        fetch_time = (time.time() - fetch_start) * 1000
        print(f"   Fetched data for {len(users_data)} users in {fetch_time:.2f}ms")
        print(f"   API calls made: {self.stats['api_calls']}")
        print(f"   Cache hits: {self.stats['cache_hits']}")
        print(
            f"   Cache hit rate: {self.stats['cache_hits']/max(self.stats['total_requests'],1):.1%}"
        )

        # Step 2: Optimize large response
        print("\n[2] Optimizing response context...")

        # Convert to tool results for context optimization
        user_results = [
            ToolResult(
                tool_id="user_data",
                success=True,
                data=user.model_dump(),
                error=None,
                execution_time_ms=fetch_time,
                timestamp=datetime.now(),
                token_estimate=len(str(user.model_dump())) // 4,
            )
            for user in users_data
        ]

        context = ContextData(tool_results=user_results)
        optimized = self.context_manager.optimize(context)

        if optimized.compression_ratio < 1.0:
            print(
                f"   Compressed: {optimized.original_tokens} → {optimized.optimized_tokens} tokens"
            )
            print(f"   Compression: {(1-optimized.compression_ratio)*100:.1f}%")
            print(f"   Method: {optimized.method_used.value}")
        else:
            print(f"   No compression needed ({optimized.original_tokens} tokens)")

        # Step 3: Build dashboard
        total_time = (time.time() - start_time) * 1000

        dashboard = DashboardData(
            users=users_data,
            total_users=len(users_data),
            aggregation_time_ms=total_time,
            cache_hits=self.stats["cache_hits"],
            api_calls=self.stats["api_calls"],
        )

        # Step 4: Serialize efficiently
        print("\n[3] Serializing dashboard...")
        serialization_request = SerializationRequest(data=dashboard)
        serialized = self.serializer.serialize(serialization_request)

        print(f"   Format: {serialized.format_used.value}")
        print(f"   Original tokens: {serialized.original_tokens}")
        print(f"   Optimized tokens: {serialized.optimized_tokens}")
        print(f"   Savings: {serialized.savings_ratio:.1%}")

        print(f"\n{'='*60}")
        print("Dashboard Complete!")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Users: {len(users_data)}")
        print(f"  Cache efficiency: {self.stats['cache_hits']}/{self.stats['total_requests']} hits")
        print("=" * 60 + "\n")

        return dashboard


# ============================================================================
# Example Usage
# ============================================================================


async def main():
    """Demonstrate API aggregation with caching."""
    print("=" * 60)
    print("API Aggregation Example with NOCP")
    print("=" * 60)

    aggregator = APIAggregator()

    # Example 1: Fetch single user (cold cache)
    print("\n\n=== Example 1: Single User (Cold Cache) ===")
    user_ids = ["user_001"]
    dashboard1 = await aggregator.fetch_dashboard(user_ids)

    # Example 2: Fetch same user again (warm cache)
    print("\n\n=== Example 2: Same User (Warm Cache) ===")
    dashboard2 = await aggregator.fetch_dashboard(user_ids)

    # Example 3: Fetch multiple users (mixed cache)
    print("\n\n=== Example 3: Multiple Users (Mixed Cache) ===")
    user_ids = ["user_001", "user_002", "user_003", "user_004", "user_005"]
    dashboard3 = await aggregator.fetch_dashboard(user_ids)

    # Example 4: Fetch many users concurrently
    print("\n\n=== Example 4: Many Users Concurrently ===")
    user_ids = [f"user_{i:03d}" for i in range(1, 21)]  # 20 users
    dashboard4 = await aggregator.fetch_dashboard(user_ids)

    # Final statistics
    print("\n\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"  Total API requests: {aggregator.stats['total_requests']}")
    print(f"  Actual API calls: {aggregator.stats['api_calls']}")
    print(f"  Cache hits: {aggregator.stats['cache_hits']}")
    print(
        f"  Overall cache hit rate: {aggregator.stats['cache_hits']/max(aggregator.stats['total_requests'],1):.1%}"
    )

    # Cache stats
    cache_stats = aggregator.cache.stats()
    print("\nCache Statistics:")
    print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Evictions: {cache_stats['evictions']}")

    print("\n" + "=" * 60)
    print("API Aggregation Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
