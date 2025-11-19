# Deployment and Operations Guide

## 1. Development Setup

### 1.1 Prerequisites

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows (via WSL)
- **Memory**: Minimum 2GB RAM
- **Disk Space**: 500MB for dependencies

### 1.2 Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd nocp

# Bootstrap (installs uv automatically if needed)
./nocp setup

# Verify installation
./nocp --version
```

The `./nocp setup` command will:
1. Check for uv installation
2. Install uv to `~/.local/bin/` if missing
3. Create virtual environment via uv
4. Install all project dependencies
5. Verify installation

### 1.3 Project Structure

```
nocp/
├── nocp                    # Executable entry point (shell script)
├── pyproject.toml         # Project metadata and dependencies
├── README.md              # User documentation
├── docs/                  # Specification documents
│   ├── 00-PROJECT-OVERVIEW.md
│   ├── 01-ARCHITECTURE.md
│   ├── 02-API-CONTRACTS.md
│   ├── 03-DEVELOPMENT-ROADMAP.md
│   ├── 04-COMPONENT-SPECS.md
│   ├── 05-TESTING-STRATEGY.md
│   └── 06-DEPLOYMENT.md (this file)
├── src/
│   └── nocp/
│       ├── __init__.py
│       ├── __main__.py        # Python entry point
│       ├── bootstrap.py       # uv auto-installer
│       ├── cli.py            # CLI interface
│       ├── agent.py          # Main orchestrator
│       ├── core/
│       │   ├── __init__.py
│       │   ├── act.py        # Tool Executor
│       │   ├── assess.py     # Context Manager
│       │   └── articulate.py # Output Serializer
│       ├── models/
│       │   ├── __init__.py
│       │   └── contracts.py  # Pydantic models
│       ├── serializers/
│       │   ├── __init__.py
│       │   └── toon.py       # TOON implementation
│       ├── llm/
│       │   ├── __init__.py
│       │   └── client.py     # LiteLLM wrapper
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── logging.py    # Structured logging
│       │   └── metrics.py    # Metrics collection
│       ├── exceptions.py     # Custom exceptions
│       ├── config.py         # Configuration management
│       └── utils/
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # pytest fixtures
│   ├── core/
│   │   ├── test_act.py
│   │   ├── test_assess.py
│   │   └── test_articulate.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── e2e/
│       └── test_agent.py
├── benchmarks/
│   ├── __init__.py
│   ├── run_benchmarks.py
│   └── datasets/
│       ├── small.json
│       ├── medium.json
│       └── large.json
└── examples/
    ├── basic_usage.py
    ├── rag_pipeline.py
    └── multi_turn_chat.py
```

---

## 2. Configuration Management

### 2.1 Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# LiteLLM Configuration
export LITELLM_LOG="INFO"
export LITELLM_API_BASE="https://api.openai.com/v1"  # Optional custom endpoint

# Proxy Agent Configuration
export NOCP_DEFAULT_MODEL="gemini/gemini-2.0-flash-exp"
export NOCP_STUDENT_MODEL="openai/gpt-4o-mini"
export NOCP_COMPRESSION_THRESHOLD="10000"
export NOCP_LOG_LEVEL="INFO"
export NOCP_ENABLE_METRICS="true"
```

### 2.2 Configuration File

Create `.nocp/config.yaml` for persistent configuration:

```yaml
# .nocp/config.yaml

llm:
  default_model: "gemini/gemini-2.0-flash-exp"
  fallback_model: "openai/gpt-4o-mini"
  student_summarizer: "openai/gpt-4o-mini"
  timeout_seconds: 60

optimization:
  compression_threshold_tokens: 10000
  target_compression_ratio: 0.40
  enable_cache: true
  cache_ttl_seconds: 3600

toon:
  threshold_array_size: 5
  include_length_markers: true

performance:
  max_concurrent_tools: 5
  request_timeout_seconds: 60

observability:
  enable_logging: true
  log_level: "INFO"
  log_format: "json"
  enable_metrics: true
  metrics_export_interval: 60
```

### 2.3 Configuration Validation

The ProxyConfig class includes Pydantic validators that catch configuration errors at startup. These validators ensure your configuration is valid before the application starts, preventing runtime failures.

#### Validation Rules

**Compression Threshold** (`default_compression_threshold`):
- **Minimum**: 1000 tokens (enforced)
- **Warning**: Values > 100,000 tokens may rarely trigger compression
- **Example Error**: `default_compression_threshold (500) is too low. Minimum recommended: 1000 tokens`

**Compression Cost Multiplier** (`compression_cost_multiplier`):
- **Minimum**: 1.0 (enforced)
- **Warning**: Values > 10.0 may reject beneficial compression
- **Rationale**: Values < 1.0 would accept compression even when it increases cost
- **Example Error**: `compression_cost_multiplier must be >= 1.0, got 0.5`

**TOON Fallback Threshold** (`toon_fallback_threshold`):
- **Range**: 0.0 to 1.0 (enforced)
- **Purpose**: Tabularity threshold below which to fallback to compact JSON
- **Example Error**: `toon_fallback_threshold must be 0.0-1.0, got 1.5`

**Log Level** (`log_level`):
- **Valid Values**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Auto-normalized**: Lowercase values are converted to uppercase
- **Example Error**: `log_level must be one of {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}, got 'INVALID'`

**Student Summarizer Max Tokens** (`student_summarizer_max_tokens`):
- **Minimum**: 100 tokens (enforced)
- **Warning**: Values > 10,000 tokens may reduce compression effectiveness
- **Example Error**: `student_summarizer_max_tokens (50) is too low. Minimum recommended: 100 tokens`

**Token Limits** (`max_input_tokens`, `max_output_tokens`):
- **Minimum**: Must be positive (> 0)
- **Warning**: Values > 10,000,000 tokens trigger a warning to verify model capabilities
- **Example Error**: `Token limit must be positive, got -1000`

#### Cross-Field Validation

The configuration also validates relationships between fields:

1. **Output vs Input Tokens**:
   - **Warning**: If `max_output_tokens` > `max_input_tokens`
   - **Reason**: May cause issues with some models

2. **Compression Threshold vs Max Input**:
   - **Error**: If `default_compression_threshold` > `max_input_tokens`
   - **Reason**: Compression would never trigger

3. **Student Summarizer vs Compression Threshold**:
   - **Warning**: If `student_summarizer_max_tokens` > `default_compression_threshold`
   - **Reason**: Student summarizer may produce outputs larger than compression trigger

#### Example Validation Errors

```bash
# Invalid compression threshold (too low)
$ export NOCP_DEFAULT_COMPRESSION_THRESHOLD=500
$ ./nocp run examples/basic_usage.py
ValidationError: default_compression_threshold (500) is too low. Minimum recommended: 1000 tokens

# Invalid TOON threshold (out of range)
$ export NOCP_TOON_FALLBACK_THRESHOLD=1.5
$ ./nocp run examples/basic_usage.py
ValidationError: toon_fallback_threshold must be 0.0-1.0, got 1.5

# Invalid cross-field configuration
$ export NOCP_MAX_INPUT_TOKENS=50000
$ export NOCP_DEFAULT_COMPRESSION_THRESHOLD=100000
$ ./nocp run examples/basic_usage.py
ValidationError: default_compression_threshold (100,000) exceeds max_input_tokens (50,000). Compression would never trigger.
```

#### Best Practices

- **Start with defaults**: The default configuration passes all validators
- **Test configuration changes**: Use `./nocp validate-config` to check configuration without running the application
- **Monitor warnings**: Pay attention to validation warnings in logs - they indicate potentially problematic settings
- **Adjust incrementally**: Make small changes to configuration values and test the impact

---

## 3. Running the Application

### 3.1 Basic Usage

```python
# examples/basic_usage.py

from nocp.agent import HighEfficiencyProxyAgent, ProxyRequest
from nocp.models.contracts import ToolRequest, ToolType

# Initialize agent
agent = HighEfficiencyProxyAgent()

# Register a custom tool
@agent.register_tool("fetch_data")
def fetch_data(query: str) -> list:
    # Your data fetching logic
    return [{"id": 1, "data": "example"}]

# Create request
request = ProxyRequest(
    query="Analyze the data",
    required_tools=[
        ToolRequest(
            tool_id="fetch_data",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="fetch_data",
            parameters={"query": "sample"}
        )
    ],
    enable_compression=True,
    enable_toon=True
)

# Process request
response = agent.process(request)

# Print results
print(f"Result: {response.result.serialized_text}")
print(f"Cost Savings: ${response.cost_analysis.savings:.4f}")
print(f"Latency: {response.total_latency_ms:.0f}ms")
print(f"Compression: {response.context_optimization.compression_ratio:.2%}")
```

### 3.2 Command-Line Interface

```bash
# Run a script
./nocp run examples/basic_usage.py

# Run with configuration override
./nocp run --model "anthropic/claude-3-5-sonnet-20241022" examples/basic_usage.py

# Run with debugging
./nocp run --debug examples/basic_usage.py

# Interactive mode
./nocp shell
```

---

## 4. Monitoring and Observability

### 4.1 Structured Logging

All transactions are logged in JSON format:

```json
{
  "timestamp": "2025-11-18T10:30:00Z",
  "request_id": "req_abc123",
  "level": "INFO",
  "message": "Request processed successfully",
  "metrics": {
    "raw_input_tokens": 15000,
    "optimized_input_tokens": 5000,
    "input_compression_ratio": 0.33,
    "llm_input_tokens": 5000,
    "llm_output_tokens": 500,
    "raw_output_tokens": 1000,
    "optimized_output_tokens": 600,
    "output_compression_ratio": 0.60,
    "total_latency_ms": 1250,
    "compression_overhead_ms": 200,
    "estimated_cost_baseline": 0.0160,
    "estimated_cost_optimized": 0.0055,
    "cost_savings": 0.0105
  },
  "model_used": "gemini/gemini-2.0-flash-exp",
  "compression_method": "semantic_pruning",
  "serialization_format": "toon"
}
```

### 4.2 Metrics Collection

Key metrics tracked:

| Metric | Type | Description |
|--------|------|-------------|
| `input_compression_ratio` | Gauge | optimized_tokens / original_tokens |
| `output_compression_ratio` | Gauge | optimized_tokens / standard_json_tokens |
| `compression_latency_ms` | Histogram | Time spent on compression |
| `llm_latency_ms` | Histogram | LLM inference time |
| `total_latency_ms` | Histogram | End-to-end request latency |
| `cost_savings_usd` | Counter | Cumulative cost savings |
| `requests_total` | Counter | Total requests processed |
| `compression_errors` | Counter | Failed compression attempts |

### 4.3 Viewing Logs and Metrics

**Current MVP Status**: The CLI commands below are not yet implemented. For MVP, use standard Python logging.

**MVP Alternative**:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

### 4.4 Future Features (Phase 3)

The following commands are planned for Phase 3 (Production Readiness):

```bash
# View real-time logs
./nocp logs --follow

# View specific time range
./nocp logs --since "2025-11-18 10:00" --until "2025-11-18 12:00"

# Filter by level
./nocp logs --level ERROR

# Export metrics
./nocp metrics export --format json > metrics.json

# View dashboard (if configured)
./nocp dashboard
```

---

## 5. Performance Tuning

### 5.1 Compression Optimization

**Adjust compression threshold:**

```python
# For most use cases: compress inputs >10k tokens
agent = HighEfficiencyProxyAgent(
    config={"compression_threshold_tokens": 10_000}
)

# For aggressive optimization: compress everything >5k tokens
agent = HighEfficiencyProxyAgent(
    config={"compression_threshold_tokens": 5_000}
)

# For minimal latency: only compress very large inputs
agent = HighEfficiencyProxyAgent(
    config={"compression_threshold_tokens": 50_000}
)
```

**Target compression ratio:**

```python
# More aggressive compression (may lose some detail)
agent = HighEfficiencyProxyAgent(
    config={"target_compression_ratio": 0.30}  # 70% reduction
)

# Conservative compression (preserves more detail)
agent = HighEfficiencyProxyAgent(
    config={"target_compression_ratio": 0.50}  # 50% reduction
)
```

### 5.2 Caching Configuration

```python
# Enable in-memory cache
agent = HighEfficiencyProxyAgent(
    config={
        "enable_cache": True,
        "cache_ttl_seconds": 3600  # 1 hour
    }
)

# For distributed systems: use Redis
agent = HighEfficiencyProxyAgent(
    config={
        "cache_backend": "redis",
        "redis_url": "redis://localhost:6379/0"
    }
)
```

### 5.3 Concurrency Settings

```python
# Adjust concurrent tool execution
agent = HighEfficiencyProxyAgent(
    config={
        "max_concurrent_tools": 10  # Up to 10 tools in parallel
    }
)
```

---

## 6. Production Deployment

### 6.1 Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . /app

# Install uv (will be done by bootstrap)
RUN ./nocp setup

# Expose health check endpoint
HEALTHCHECK --interval=30s --timeout=3s \
  CMD ./nocp health || exit 1

# Run application
# NOTE: The following CMD is a non-functional placeholder for demonstration purposes only.
# This Dockerfile example shows the structure, but the CMD needs to be replaced based on your use case:
#   - For API server: CMD ["./nocp", "serve"] (requires implementing serve command)
#   - For development: CMD ["./nocp", "shell"] (interactive shell)
#   - For job processing: Implement a custom worker command
# The 'info' command only displays project information and exits immediately.
CMD ["./nocp", "info"]  # Placeholder - replace with actual application entry point
```

**Build and run:**

```bash
# Build image
docker build -t nocp:latest .

# Run container
docker run -d \
  -e OPENAI_API_KEY="sk-..." \
  -e GOOGLE_API_KEY="..." \
  -p 8000:8000 \
  --name nocp-agent \
  nocp:latest
```

### 6.2 Kubernetes Deployment

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nocp-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nocp-agent
  template:
    metadata:
      labels:
        app: nocp-agent
    spec:
      containers:
      - name: nocp
        image: nocp:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: openai-api-key
        - name: NOCP_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 6.3 Scaling Considerations

**Horizontal Scaling:**
- Stateless design allows multiple replicas
- Use Redis for shared caching layer
- Load balancer distributes requests

**Vertical Scaling:**
- Increase memory for larger context windows
- More CPU cores enable higher concurrency

**Cost Optimization:**
- Monitor compression effectiveness
- Adjust thresholds based on actual savings
- Use cheaper models for summarization

---

## 7. Troubleshooting

### 7.1 Common Issues

**Issue: uv installation fails**

```bash
# Manual fallback
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
./nocp setup
```

**Issue: Compression not reducing tokens**

```bash
# For MVP, use Python logging to debug (Phase 3 will add ./nocp logs command)
# Enable debug mode when running
./nocp run --debug examples/basic_usage.py

# Adjust threshold
export NOCP_COMPRESSION_THRESHOLD="5000"
```

**Issue: High latency overhead**

```bash
# Benchmark compression cost
./nocp benchmark --component assess

# Disable compression if overhead too high
export NOCP_ENABLE_COMPRESSION="false"
```

### 7.2 Debug Mode

```bash
# Enable debug logging
./nocp run --debug examples/basic_usage.py

# This will show:
# - Token counts at each stage
# - Compression method selection
# - Cost-benefit calculations
# - Serialization format negotiation
```

### 7.3 Health Checks

```bash
# Check system health
./nocp health

# Output:
# ✓ uv installation: OK
# ✓ Dependencies: OK
# ✓ LLM connectivity: OK (gemini/gemini-2.0-flash-exp)
# ✓ Student model: OK (openai/gpt-4o-mini)
# ✓ Cache: OK (in-memory)
```

---

## 8. Security Best Practices

### 8.1 API Key Management

- **Never commit API keys** to version control
- Use environment variables or secret management systems
- Rotate keys regularly
- Use separate keys for dev/staging/production

### 8.2 Input Validation

- All external inputs validated via Pydantic models
- Rate limiting on API endpoints
- Sanitize user-provided tool parameters

### 8.3 Dependency Management

```bash
# Audit dependencies (Phase 3 - not yet implemented)
# For MVP, use uv directly:
uv pip list
uv pip check

# Update dependencies (Phase 3 - not yet implemented)
# For MVP, edit pyproject.toml and re-run:
./nocp setup

# Pin versions in production
# (Already handled by uv's lockfile when using uv sync)
```

---

## 9. Backup and Recovery

### 9.1 Configuration Backup

```bash
# Backup configuration
./nocp config export > nocp-config-backup.yaml

# Restore configuration
./nocp config import < nocp-config-backup.yaml
```

### 9.2 Cache Management

```bash
# Clear cache
./nocp cache clear

# Export cache for migration
./nocp cache export > cache-dump.json

# Import cache
./nocp cache import < cache-dump.json
```

---

## 10. Maintenance

### 10.1 Regular Tasks

- **Weekly**: Review metrics dashboard
- **Monthly**: Update dependencies
- **Quarterly**: Audit API key usage
- **Annually**: Review compression strategies

### 10.2 Monitoring Alerts

Set up alerts for:
- Compression ratio degradation (>0.70)
- High latency (>2x baseline)
- Error rate spike (>5%)
- Cost anomalies

---

## 11. Release Process

### 11.1 Version Management

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### 11.2 Changelog

Maintain `CHANGELOG.md`:

```markdown
# Changelog

## [0.2.0] - 2025-11-25

### Added
- Conversation history compaction
- Redis caching support
- Async tool execution

### Fixed
- Token estimation accuracy
- TOON encoding for nested structures

### Changed
- Default compression threshold: 5000 → 10000 tokens
```

---

## 12. Support and Resources

### 12.1 Documentation

- **API Reference**: Generated from docstrings
- **User Guide**: `docs/00-PROJECT-OVERVIEW.md`
- **Examples**: `examples/` directory

### 12.2 Community

- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Contributing**: See `CONTRIBUTING.md`

---

**Document Version**: 1.0
**Last Updated**: 2025-11-18
**Next Review**: 2025-12-18
