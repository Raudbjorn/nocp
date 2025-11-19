"""
Advanced Example: RAG (Retrieval-Augmented Generation) with NOCP

This example demonstrates how to build a production-ready RAG system using NOCP
to optimize token usage and improve performance.

Features:
- Vector similarity search with caching
- Semantic pruning of retrieved chunks
- Context optimization for LLM
- Async execution for better performance
"""

import asyncio
import hashlib
import os
import sys
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pydantic import BaseModel, Field

from nocp.core.act import ToolExecutor
from nocp.core.assess import ContextManager
from nocp.core.articulate import OutputSerializer
from nocp.core.cache import LRUCache
from nocp.core.async_modules import ConcurrentToolExecutor
from nocp.models.contracts import (
    ToolRequest,
    ToolType,
    ContextData,
    ToolResult,
    SerializationRequest
)


# ============================================================================
# Data Models
# ============================================================================

class DocumentChunk(BaseModel):
    """A chunk of text from a document."""
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)
    similarity_score: float = 0.0


class RAGQuery(BaseModel):
    """RAG query request."""
    query: str
    top_k: int = 5
    min_similarity: float = 0.5


class RAGResponse(BaseModel):
    """RAG response with sources."""
    answer: str
    sources: List[DocumentChunk]
    total_tokens: int
    optimized_tokens: int


# ============================================================================
# Simulated Vector Database
# ============================================================================

class VectorDB:
    """Simulated vector database for demonstration."""

    def __init__(self):
        # In production, this would be Pinecone, Weaviate, Qdrant, etc.
        self.documents = [
            DocumentChunk(
                id="doc_1",
                text="NOCP is a high-efficiency proxy agent designed to minimize token usage in LLM applications.",
                metadata={"source": "docs/overview.md", "page": 1}
            ),
            DocumentChunk(
                id="doc_2",
                text="The Act-Assess-Articulate pipeline optimizes tokens at every stage of execution.",
                metadata={"source": "docs/architecture.md", "page": 3}
            ),
            DocumentChunk(
                id="doc_3",
                text="Caching can improve performance by 10-100x for repeated queries.",
                metadata={"source": "docs/performance.md", "page": 5}
            ),
            DocumentChunk(
                id="doc_4",
                text="Context compression reduces token usage by 50-70% using semantic pruning.",
                metadata={"source": "docs/optimization.md", "page": 2}
            ),
            DocumentChunk(
                id="doc_5",
                text="NOCP supports both synchronous and asynchronous execution modes.",
                metadata={"source": "docs/api.md", "page": 7}
            ),
            DocumentChunk(
                id="doc_6",
                text="The TOON serialization format can save 30-60% on output tokens for tabular data.",
                metadata={"source": "docs/serialization.md", "page": 4}
            ),
            DocumentChunk(
                id="doc_7",
                text="Redis caching enables distributed caching across multiple servers.",
                metadata={"source": "docs/deployment.md", "page": 6}
            ),
            DocumentChunk(
                id="doc_8",
                text="Concurrent tool execution allows processing multiple requests in parallel.",
                metadata={"source": "docs/concurrency.md", "page": 8}
            ),
        ]

    async def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Simulate vector similarity search.

        In production, this would:
        1. Embed the query using an embedding model
        2. Search vector database for similar documents
        3. Return top-k results with similarity scores
        """
        # Simulate embedding computation delay
        await asyncio.sleep(0.05)

        # Simple keyword matching for demonstration
        # In production, use actual embeddings (OpenAI, Cohere, etc.)
        results = []
        query_lower = query.lower()

        for doc in self.documents:
            # Calculate simple similarity score based on keyword overlap
            doc_words = set(doc.text.lower().split())
            query_words = set(query_lower.split())
            overlap = len(doc_words & query_words)
            total = len(doc_words | query_words)
            score = overlap / total if total > 0 else 0.0

            if score > 0:
                doc_copy = doc.model_copy()
                doc_copy.similarity_score = score
                results.append(doc_copy)

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]


# ============================================================================
# RAG System
# ============================================================================

class RAGSystem:
    """Production-ready RAG system with NOCP optimizations."""

    def __init__(self):
        # Initialize components
        self.vector_db = VectorDB()

        # Setup caching (improves performance for repeated queries)
        self.cache = LRUCache(max_size=1000, default_ttl=3600)

        # Setup executor with cache
        self.executor = ToolExecutor(cache=self.cache)
        self.concurrent = ConcurrentToolExecutor(self.executor, max_concurrent=5)

        # Setup context optimizer
        self.context_manager = ContextManager(
            compression_threshold=1000,  # Compress if >1000 tokens
            target_compression_ratio=0.40,  # Target 60% reduction
            enable_litellm=False  # Disable for example
        )

        # Setup output serializer
        self.serializer = OutputSerializer()

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register RAG tools."""

        @self.executor.register_async_tool("vector_search")
        async def vector_search(query: str, top_k: int = 5) -> List[dict]:
            """Search vector database for relevant documents."""
            chunks = await self.vector_db.search(query, top_k)
            return [chunk.model_dump() for chunk in chunks]

        @self.executor.register_async_tool("rerank")
        async def rerank(query: str, chunks: List[dict], top_k: int = 3) -> List[dict]:
            """
            Rerank chunks using a more sophisticated model.

            In production, this would use a cross-encoder model
            for better relevance ranking.
            """
            # Simulate reranking delay
            await asyncio.sleep(0.03)

            # Simple reranking: boost chunks with query terms
            query_words = set(query.lower().split())

            for chunk in chunks:
                text_words = set(chunk['text'].lower().split())
                boost = len(query_words & text_words) * 0.1
                chunk['similarity_score'] += boost

            # Re-sort and take top-k
            chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            return chunks[:top_k]

        @self.executor.register_async_tool("generate_answer")
        async def generate_answer(query: str, context: str) -> str:
            """
            Generate answer using LLM.

            In production, this would call an actual LLM API
            (OpenAI, Anthropic, Google, etc.)
            """
            # Simulate LLM call delay
            await asyncio.sleep(0.1)

            # Simulated response
            return f"Based on the provided context, {query.lower()} - [Answer generated from context: {context[:100]}...]"

    async def query(self, query_text: str, top_k: int = 5) -> RAGResponse:
        """
        Process a RAG query end-to-end with optimization.

        Steps:
        1. Vector search for relevant chunks
        2. Rerank chunks for better relevance
        3. Optimize context to reduce tokens
        4. Generate answer using LLM
        5. Serialize response efficiently
        """
        print(f"\n{'='*60}")
        print(f"Processing RAG Query: '{query_text}'")
        print('='*60)

        # Step 1: Vector search
        print("\n[1] Performing vector search...")
        search_request = ToolRequest(
            tool_id="vector_search",
            tool_type=ToolType.RAG_RETRIEVAL,
            function_name="vector_search",
            parameters={"query": query_text, "top_k": top_k}
        )

        search_result = await self.executor.execute_async(search_request)
        chunks = search_result.data
        print(f"   Found {len(chunks)} relevant chunks")

        # Step 2: Rerank
        print("\n[2] Reranking chunks...")
        rerank_request = ToolRequest(
            tool_id="rerank",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="rerank",
            parameters={"query": query_text, "chunks": chunks, "top_k": 3}
        )

        rerank_result = await self.executor.execute_async(rerank_request)
        reranked_chunks = rerank_result.data
        print(f"   Top {len(reranked_chunks)} chunks after reranking")

        # Step 3: Optimize context
        print("\n[3] Optimizing context...")

        # Convert chunks to ToolResult for context optimization
        chunk_results = [
            ToolResult(
                tool_id="vector_search",
                success=True,
                data=chunk,
                error=None,
                execution_time_ms=search_result.execution_time_ms,
                timestamp=datetime.now(),
                token_estimate=len(chunk['text']) // 4
            )
            for chunk in reranked_chunks
        ]

        context = ContextData(
            tool_results=chunk_results,
            transient_context={"query": query_text}
        )

        optimized = self.context_manager.optimize(context)
        print(f"   Original tokens: {optimized.original_tokens}")
        print(f"   Optimized tokens: {optimized.optimized_tokens}")
        print(f"   Compression ratio: {optimized.compression_ratio:.2%}")
        print(f"   Method: {optimized.method_used.value}")

        # Step 4: Generate answer
        print("\n[4] Generating answer...")
        generate_request = ToolRequest(
            tool_id="generate_answer",
            tool_type=ToolType.API_CALL,
            function_name="generate_answer",
            parameters={
                "query": query_text,
                "context": optimized.optimized_text
            }
        )

        answer_result = await self.executor.execute_async(generate_request)
        answer = answer_result.data
        print(f"   Answer generated ({answer_result.execution_time_ms:.2f}ms)")

        # Step 5: Build response
        print("\n[5] Building response...")
        source_chunks = [
            DocumentChunk(
                id=chunk['id'],
                text=chunk['text'],
                metadata=chunk['metadata'],
                similarity_score=chunk['similarity_score']
            )
            for chunk in reranked_chunks
        ]

        response = RAGResponse(
            answer=answer,
            sources=source_chunks,
            total_tokens=optimized.original_tokens,
            optimized_tokens=optimized.optimized_tokens
        )

        # Serialize response efficiently
        serialization_request = SerializationRequest(data=response)
        serialized = self.serializer.serialize(serialization_request)

        print(f"\n[6] Response serialization:")
        print(f"   Format: {serialized.format_used.value}")
        print(f"   Savings: {serialized.savings_ratio:.1%}")

        # Cache statistics
        cache_stats = self.cache.stats()
        print(f"\n[7] Cache statistics:")
        print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"   Cache size: {cache_stats['size']}/{cache_stats['max_size']}")

        print(f"\n{'='*60}\n")

        return response


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Demonstrate RAG system with multiple queries."""
    print("="*60)
    print("RAG System Example with NOCP")
    print("="*60)

    # Create RAG system
    rag = RAGSystem()

    # Example queries
    queries = [
        "How does NOCP optimize tokens?",
        "What caching options are available?",
        "How do I use async execution?",
        "How does NOCP optimize tokens?",  # Repeat to show caching
    ]

    # Process queries
    for i, query in enumerate(queries, 1):
        print(f"\n\nQuery {i}/{len(queries)}")

        response = await rag.query(query, top_k=5)

        print(f"\n✓ Answer:")
        print(f"  {response.answer}")

        print(f"\n✓ Sources ({len(response.sources)}):")
        for j, source in enumerate(response.sources, 1):
            print(f"  {j}. {source.text[:60]}...")
            print(f"     Similarity: {source.similarity_score:.2f}")
            print(f"     Source: {source.metadata.get('source', 'unknown')}")

        print(f"\n✓ Token Optimization:")
        print(f"  Original: {response.total_tokens} tokens")
        print(f"  Optimized: {response.optimized_tokens} tokens")
        print(f"  Savings: {(1 - response.optimized_tokens/response.total_tokens)*100:.1f}%")

        # Small delay between queries
        if i < len(queries):
            await asyncio.sleep(0.5)

    print("\n" + "="*60)
    print("RAG Example Complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
