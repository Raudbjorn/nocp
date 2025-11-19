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
    {"id": "1", "name": "Alice", "email": "alice@ex.com", "created": "2024-01-01", "updated": "2024-01-01"},
    {"id": "2", "name": "Bob", "email": "bob@ex.com", "created": "2024-01-01", "updated": "2024-01-01"}
  ]
}
```
Note: The full list contains 100 more similar user objects.

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

NOCP automatically calculates whether compression is worthwhile based on token efficiency:

```python
# The calculus is based on token counts, not monetary cost.
# compression_cost is the number of tokens used by the student model.
compression_cost = tokens_used_by_student_model
net_token_savings = original_tokens - compressed_tokens - compression_cost

# The decision to use the compressed version depends on whether the
# net savings are significant enough, determined by a multiplier.
justification_threshold = compression_cost * config.compression_cost_multiplier

if net_token_savings > justification_threshold:
    # Compress - the token savings justify the compression overhead
else:
    # Use original - compression doesn't provide enough benefit
```

The `compression_cost_multiplier` (default: 2.0) ensures compression provides meaningful savings beyond just breaking even. For example, with a multiplier of 2.0, the net token savings must be at least twice the tokens spent on compression.

## Combining Strategies

Strategies can be combined for maximum effect:

```python
config = ProxyConfig(
    enable_semantic_pruning=True,      # Remove redundancy
    enable_knowledge_distillation=True, # Summarize verbose outputs
    enable_history_compaction=True,     # Compress conversation history
)
```

**How strategies are applied**:

These strategies operate on different parts of the request context and are not necessarily applied sequentially to the same content:

- **Semantic Pruning** and **Knowledge Distillation** operate on tool outputs when they exceed configured thresholds
- **History Compaction** operates on conversation history to manage multi-turn interactions

Within tool output compression, when both are enabled:
1. Semantic Pruning is applied first (structural optimization)
2. Knowledge Distillation may be applied to the result (content summarization)

## Best Practices

### 1. Tool-Specific Thresholds

Set different thresholds for different tools:

```python
config = ProxyConfig()
config.register_tool_threshold("database_query", 10_000)
config.register_tool_threshold("web_search", 5_000)
```

### 2. Monitor Compression Ratios

Track compression effectiveness:

```python
response, metrics = agent.process_request(request)

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

Compare compressed vs uncompressed by using different configurations:

```python
# Create agent with compression enabled
compressed_config = ProxyConfig(
    enable_semantic_pruning=True,
    enable_knowledge_distillation=True
)
compressed_agent = NOCPAgent(compressed_config)

# Create agent with compression disabled
uncompressed_config = ProxyConfig(
    enable_semantic_pruning=False,
    enable_knowledge_distillation=False
)
uncompressed_agent = NOCPAgent(uncompressed_config)

# Process the same request with both configurations
compressed_response, compressed_metrics = compressed_agent.process_request(request)
uncompressed_response, uncompressed_metrics = uncompressed_agent.process_request(request)

# Compare quality and efficiency
print(f"Compressed quality: {evaluate_quality(compressed_response)}")
print(f"Uncompressed quality: {evaluate_quality(uncompressed_response)}")
print(f"Token savings: {uncompressed_metrics.raw_input_tokens - compressed_metrics.compressed_input_tokens:,}")
```

## Troubleshooting

### Low Compression Ratios

**Problem**: Compression ratio >70% (less than 30% reduction)

**Solutions**:
- Lower compression threshold
- Enable additional strategies
- Check data for actual redundancy
- Decrease target compression ratio (make compression more aggressive)

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
    transaction_id: str                          # Unique identifier for this request
    raw_input_tokens: int                        # Before compression
    compressed_input_tokens: int                 # After compression
    input_compression_ratio: float               # Compressed / Raw (e.g., 0.7 = 30% reduction)

    raw_output_tokens: int                       # Output tokens from LLM
    final_output_format: str                     # Format of the final output
    output_token_savings: int                    # Tokens saved in output

    total_latency_ms: float                      # Total request time
    compression_latency_ms: float                # Time spent compressing
    llm_inference_latency_ms: float              # Time spent in LLM

    tools_used: List[str]                        # Which tools were called
    compression_operations: List[CompressionResult]  # Detailed compression info
```

Each `CompressionResult` contains:
- `compression_method`: The strategy used (e.g., "semantic_pruning", "knowledge_distillation")
- `original_size`: Tokens before compression
- `compressed_size`: Tokens after compression
- `compression_ratio`: Ratio of compressed to original

Access via:

```python
response, metrics = agent.process_request(request)

print(f"Input token savings: {metrics.raw_input_tokens - metrics.compressed_input_tokens:,}")
print(f"Compression ratio: {metrics.input_compression_ratio:.0%}")
print(f"Strategies used: {', '.join([op.compression_method for op in metrics.compression_operations])}")

# Access detailed compression information
for op in metrics.compression_operations:
    print(f"  {op.compression_method}: {op.original_size} -> {op.compressed_size} tokens ({op.compression_ratio:.0%})")
```
