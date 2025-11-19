# Claude Code Developer Guide for NOCP

This file provides guidance to Claude Code and other AI assistants when working with the NOCP codebase.

## Quick Start

### Development Setup
```bash
# Clone and enter directory
git clone https://github.com/your-org/nocp.git && cd nocp

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python tests/run_tests.py
pytest tests/unit/  # Unit tests only
```

### Key Commands

```bash
# Code quality
black src/nocp tests/              # Format code
ruff check src/nocp tests/         # Lint
mypy src/nocp                      # Type check

# Testing
pytest                             # All tests
pytest tests/unit/                 # Unit tests only
pytest tests/integration/          # Integration tests
pytest --cov=nocp                  # With coverage
python tests/run_tests.py          # Custom runner

# Pre-commit
pre-commit run --all-files         # Run all hooks
```

## Project Structure

```
nocp/
├── src/nocp/
│   ├── core/              # Core architecture components
│   │   ├── agent.py       # Main orchestrator (HighEfficiencyProxyAgent)
│   │   ├── act.py         # Tool execution logic (legacy, use modules/ instead)
│   │   ├── assess.py      # Context compression logic
│   │   ├── articulate.py  # Output serialization logic
│   │   ├── config.py      # Configuration (ProxyConfig)
│   │   └── persistence.py # State persistence
│   ├── modules/           # Reusable implementation modules (ACTIVE)
│   │   ├── context_manager.py  # ContextManager (Assess implementation)
│   │   ├── output_serializer.py # OutputSerializer (Articulate implementation)
│   │   ├── tool_executor.py    # ToolExecutor (Act implementation)
│   │   └── router.py           # RequestRouter
│   ├── models/            # Pydantic schemas
│   │   ├── schemas.py     # Request/response models
│   │   ├── contracts.py   # Tool contracts
│   │   ├── context.py     # Context models
│   │   └── enums.py       # Configuration enums
│   ├── llm/               # LLM integration
│   │   ├── client.py      # LiteLLM wrapper
│   │   └── router.py      # Model routing
│   ├── serializers/       # Output serialization formats
│   │   └── toon.py        # TOON format implementation
│   ├── observability/     # Monitoring and logging
│   │   └── logging.py     # Enhanced logging utilities
│   ├── utils/             # Utilities
│   │   ├── logging.py     # Structured logging
│   │   └── token_counter.py
│   └── tools/             # Example tools
├── tests/
│   ├── unit/              # Fast, isolated tests
│   ├── integration/       # Component integration
│   ├── e2e/               # End-to-end workflows
│   └── performance/       # Benchmarks
├── docs/                  # Architecture and specs
├── examples/              # Demo scripts
└── benchmarks/            # Performance benchmarks
```

**Important**: The `core/` directory contains both high-level orchestration (`agent.py`, `config.py`) and some legacy implementations. The `modules/` directory contains the **active implementations** used by the agent. When developing:
- Use `modules/tool_executor.py` for tool execution (not `core/act.py`)
- Use `modules/context_manager.py` for context management (implements logic from `core/assess.py`)
- Use `modules/output_serializer.py` for output formatting (implements logic from `core/articulate.py`)

## Architecture: Act-Assess-Articulate

NOCP implements a three-phase pipeline for efficient LLM interaction:

### 1. Act (Tool Execution)
- **Implementation**: `src/nocp/modules/tool_executor.py`
- **Logic/Design**: `src/nocp/core/act.py` (legacy reference)
- **Purpose**: Execute tools with retry logic and timeout handling
- **Key Classes**: `ToolExecutor`, `ToolRequest`, `ToolResult`
- **Models**: `src/nocp/models/contracts.py`

### 2. Assess (Context Optimization)
- **Implementation**: `src/nocp/modules/context_manager.py`
- **Logic/Design**: `src/nocp/core/assess.py`
- **Purpose**: Compress context to reduce token usage
- **Strategies**: Semantic Pruning, Knowledge Distillation, History Compaction
- **Key Classes**: `ContextManager`, `ContextData`, `OptimizedContext`
- **Models**: `src/nocp/models/context.py`

### 3. Articulate (Output Serialization)
- **Implementation**: `src/nocp/modules/output_serializer.py`
- **Logic/Design**: `src/nocp/core/articulate.py`
- **Purpose**: Serialize output efficiently (TOON or compact JSON)
- **Key Classes**: `OutputSerializer`, `SerializationRequest`
- **TOON Format**: `src/nocp/serializers/toon.py`

### Orchestrator
- **File**: `src/nocp/core/agent.py`
- **Purpose**: Coordinate all phases and handle errors
- **Key Class**: `HighEfficiencyProxyAgent`
- **Configuration**: `src/nocp/core/config.py` (`ProxyConfig`)

## Development Guidelines

### Adding a New Tool

1. Define tool function with type hints
2. Register with ToolExecutor
3. Add tool schema for Gemini/LiteLLM
4. Write unit tests

```python
# In your module
from nocp.modules.tool_executor import ToolExecutor
from pydantic import BaseModel

executor = ToolExecutor()

# Define tool schema (Pydantic model)
class MyToolParams(BaseModel):
    param: str

@executor.register_tool("my_tool")
def my_tool(param: str) -> dict:
    """Tool description"""
    return {"result": f"Processed {param}"}

# For LiteLLM/Gemini, add to tool definitions list
from nocp.models.schemas import ToolDefinition

my_tool_def = ToolDefinition(
    name="my_tool",
    description="Tool description for LLM",
    parameters=MyToolParams.model_json_schema()
)

# Tests in tests/unit/test_my_tool.py
def test_my_tool():
    result = my_tool("test")
    assert result["result"] == "Processed test"
```

### Adding a Compression Strategy

1. Add method to `ContextManager`
2. Add `enable_*` config flag in `ProxyConfig`
3. Track metrics in `ContextMetrics`
4. Add tests

```python
# In src/nocp/modules/context_manager.py
def _my_compression(self, context: ContextData) -> str:
    """Custom compression strategy"""
    # Implementation
    return compressed_text

# In src/nocp/core/config.py
class ProxyConfig(BaseSettings):
    enable_my_compression: bool = Field(default=False)
```

### Error Handling

Use the Result pattern for explicit error handling:

```python
from nocp.models.result import Result

def my_function() -> Result[MyData]:
    try:
        data = do_something()
        return Result.ok(data)
    except Exception as e:
        return Result.err(str(e))

# Usage
result = my_function()
if result.success:
    process(result.data)
else:
    logger.error(result.error)
```

### Logging

Use component-specific loggers:

```python
from nocp.utils.logging import ComponentLogger

logger = ComponentLogger("my_component")

logger.log_operation_start("process_data", {"items": 100})
# ... do work ...
logger.log_operation_complete("process_data", duration_ms=123.45)
```

### Configuration

Configuration precedence (highest to lowest):
1. CLI arguments
2. Environment variables (direct field names, e.g., `GEMINI_API_KEY`, `LOG_LEVEL`)
3. `.env` file
4. `[tool.nocp]` in `pyproject.toml`
5. Hardcoded defaults

**Note**: Environment variables do NOT use a `NOCP_` prefix. They are named directly from the field names in `ProxyConfig` (e.g., `gemini_api_key` -> `GEMINI_API_KEY`).

Example pyproject.toml:

```toml
[tool.nocp]
compression_threshold = 5000
enable_semantic_pruning = true
log_level = "INFO"
```

## Testing Strategy

### Test Categories

- **Unit** (`tests/unit/`): Fast (<100ms), isolated, mocked dependencies
- **Integration** (`tests/integration/`): Component integration, may call APIs
- **E2E** (`tests/e2e/`): Full workflows, requires API keys
- **Performance** (`tests/performance/`): Benchmarks

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific category
python tests/run_tests.py --category unit
pytest tests/unit/

# Run with coverage
pytest --cov=nocp --cov-report=html

# Run specific test
pytest tests/unit/test_config.py::test_load_from_pyproject
```

### Writing Tests

Always use fixtures from conftest.py:

```python
def test_compression(config_all_features, large_tool_output):
    """Test compression with all features enabled"""
    manager = ContextManager(config=config_all_features)
    result = manager.compress(large_tool_output)

    assert result.compression_ratio < 0.5
```

## Common Tasks

### Running Examples

```bash
# Set API key
export GEMINI_API_KEY='your-key'

# Run basic demo
python examples/demo_basic.py

# Run with custom config
NOCP_COMPRESSION_THRESHOLD=10000 python examples/demo_basic.py
```

### Debugging

Enable debug logging:

```bash
NOCP_LOG_LEVEL=DEBUG python examples/demo_basic.py
```

Or in code:

```python
from nocp.core.config import ProxyConfig

config = ProxyConfig(log_level="DEBUG")
```

### Performance Profiling

```bash
# Run with profiling
python -m cProfile -o profile.stats examples/demo_basic.py

# View results
python -m pstats profile.stats
```

## Known Issues

- Latency overhead from compression (50-200ms per request)
- TOON serialization needs more real-world validation
- Drift detection requires tuning for specific use cases
- LiteLLM response parsing varies by provider

## CI/CD

### GitHub Actions

- Linting and type checking on every PR
- Tests on Ubuntu, macOS, Windows
- Python 3.10, 3.11, 3.12
- Coverage uploaded to Codecov

### Local Pre-commit

Pre-commit hooks run automatically:
- black (formatting)
- ruff (linting)
- mypy (type checking)
- trailing whitespace, YAML/JSON validation

## Resources

- [Architecture Overview](docs/01-ARCHITECTURE.md)
- [API Contracts](docs/02-API-CONTRACTS.md)
- [Development Roadmap](docs/03-DEVELOPMENT-ROADMAP.md)
- [Component Specs](docs/04-COMPONENT-SPECS.md)
- [Testing Strategy](docs/05-TESTING-STRATEGY.md)
- [Deployment Guide](docs/06-DEPLOYMENT.md)
