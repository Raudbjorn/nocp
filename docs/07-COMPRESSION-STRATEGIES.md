# Context Compression Strategies

## Overview

NOCP implements multiple compression strategies to reduce input token usage while preserving semantic meaning. Each strategy has different characteristics and use cases.

## Available Strategies

### 1. Semantic Pruning

**Purpose**: Remove redundant information from RAG outputs and API responses.

**When to Use**:
- Tool outputs over threshold (default: 5000 tokens)
- Data contains repeated fields or verbose descriptions
- Multiple similar items in lists

**How It Works**:
1. Identify repeated patterns
2. Extract unique information
3. Deduplicate similar items
4. Preserve essential context

**Configuration**:
```bash
NOCP_ENABLE_SEMANTIC_PRUNING=true
NOCP_SEMANTIC_PRUNING_THRESHOLD=5000
```

**Performance**:
- Average compression: 60-70%
- Processing overhead: 50-100ms
- Cost savings: ~$0.03 per 10K tokens compressed

**Example**:

Before (2000 tokens):
```json
{
  "users": [
    {"id": "1", "name": "Alice", "email": "alice@ex.com", "created": "2024-01-01", "updated": "2024-01-01", ...},
    {"id": "2", "name": "Bob", "email": "bob@ex.com", "created": "2024-01-01", "updated": "2024-01-01", ...},
    // ... 100 more users
  ]
}
```

After (700 tokens):
```json
{
  "users_summary": {
    "total": 102,
    "sample": [
      {"id": "1", "name": "Alice", "email": "alice@ex.com"},
      {"id": "2", "name": "Bob", "email": "bob@ex.com"}
    ],
    "common_fields": ["created: 2024-01-01", "updated: 2024-01-01"]
  }
}
```

### 2. Knowledge Distillation

**Purpose**: Use a lightweight "student" model to summarize verbose outputs.

**When to Use**:
- Complex outputs that can be summarized
- Cost-of-Compression Calculus justifies it
- Narrative or explanatory content

**How It Works**:
1. Send verbose output to student model (e.g., Gemini 1.5 Flash 8B)
2. Request concise summary preserving key information
3. Validate compression meets target ratio
4. Fall back to original if insufficient compression

**Configuration**:
```bash
NOCP_ENABLE_KNOWLEDGE_DISTILLATION=true
NOCP_STUDENT_MODEL=gemini/gemini-1.5-flash-8b
NOCP_TARGET_COMPRESSION_RATIO=0.4  # 60% reduction
```

**Cost Analysis**:
- Student model: ~$0.15 per 1M tokens
- Must save more in main model costs than student costs
- Typically worthwhile for inputs >10K tokens

**Example**:

Before (3000 tokens):
```
[Long technical documentation with examples, code snippets, explanations, etc.]
```

After (1200 tokens):
```
Summary: [Concise overview preserving key concepts, API signatures, and critical examples]
```

### 3. History Compaction

**Purpose**: Compress conversation history for multi-turn interactions.

**When to Use**:
- Multi-turn conversations
- History exceeds context window
- Older messages less relevant

**How It Works**:
1. Identify conversation "turns" (user/assistant pairs)
2. Summarize older turns
3. Keep recent turns verbatim
4. Maintain conversation flow

**Configuration**:
```bash
NOCP_ENABLE_HISTORY_COMPACTION=true
NOCP_HISTORY_WINDOW_SIZE=5  # Keep last 5 turns verbatim
```

**Example**:

Before (8000 tokens across 20 turns):
```
Turn 1: [full conversation]
Turn 2: [full conversation]
...
Turn 20: [full conversation]
```

After (3000 tokens):
```
Earlier conversation summary: [Key decisions, context, outcomes]
Turn 16-20: [verbatim recent conversation]
```

## Cost-of-Compression Calculus

NOCP automatically calculates whether compression is worthwhile:

```python
compression_cost = (tokens_to_compress / 1_000_000) * student_model_cost_per_million
token_savings = original_tokens - compressed_tokens
main_model_savings = (token_savings / 1_000_000) * main_model_cost_per_million

if main_model_savings > compression_cost:
    # Compress
else:
    # Use original
```

## Combining Strategies

Strategies can be combined for maximum effect:

```python
config = ProxyConfig(
    enable_semantic_pruning=True,      # Remove redundancy
    enable_knowledge_distillation=True, # Summarize verbose outputs
    enable_history_compaction=True,     # Compress conversation history
)
```

**Order of operations**:
1. Semantic Pruning (structural optimization)
2. Knowledge Distillation (content summarization)
3. History Compaction (temporal optimization)

## Best Practices

### 1. Tool-Specific Thresholds

Set different thresholds for different tools:

```python
config = ProxyConfig()
config.set_tool_compression_threshold("database_query", 10_000)
config.set_tool_compression_threshold("web_search", 5_000)
```

### 2. Monitor Compression Ratios

Track compression effectiveness:

```python
metrics = agent.process_request(request)

if metrics.input_compression_ratio > 0.8:  # <20% compression
    logger.warning("Low compression ratio - consider adjusting thresholds")
```

### 3. Validate Compression Quality

For critical applications, validate compressed output preserves meaning:

```python
# Generate embedding of original
original_embedding = embed(original_text)

# Generate embedding of compressed
compressed_embedding = embed(compressed_text)

# Check similarity
similarity = cosine_similarity(original_embedding, compressed_embedding)

if similarity < 0.8:
    logger.warning("Compression may have lost important information")
```

### 4. A/B Testing

Compare compressed vs uncompressed:

```python
# Process with compression
compressed_response = agent.process(request, enable_compression=True)

# Process without compression
uncompressed_response = agent.process(request, enable_compression=False)

# Compare quality and cost
print(f"Compressed quality: {evaluate_quality(compressed_response)}")
print(f"Cost savings: {compressed_response.cost_savings}")
```

## Troubleshooting

### Low Compression Ratios

**Problem**: Compression ratio >70% (less than 30% reduction)

**Solutions**:
- Lower compression threshold
- Enable additional strategies
- Check data for actual redundancy
- Increase target compression ratio

### Over-Compression

**Problem**: Compressed output loses important information

**Solutions**:
- Increase target compression ratio (less aggressive)
- Disable aggressive strategies
- Add validation checks
- Use tool-specific thresholds

### High Latency

**Problem**: Compression adds >200ms latency

**Solutions**:
- Disable Knowledge Distillation for time-sensitive requests
- Increase compression threshold (compress less often)
- Use faster student model
- Consider async compression

## Metrics Reference

Key metrics tracked by NOCP:

```python
class ContextMetrics:
    raw_input_tokens: int              # Before compression
    compressed_input_tokens: int       # After compression
    input_compression_ratio: float     # Compressed / Raw

    compression_latency_ms: float      # Time spent compressing
    llm_inference_latency_ms: float    # Time spent in LLM

    compression_operations: List[str]  # Which strategies were used
    token_savings: int                 # Total tokens saved
```

Access via:

```python
response, metrics = agent.process_request(request)

print(f"Token savings: {metrics.raw_input_tokens - metrics.compressed_input_tokens:,}")
print(f"Compression ratio: {metrics.input_compression_ratio:.0%}")
print(f"Strategies used: {', '.join(metrics.compression_operations)}")
```
