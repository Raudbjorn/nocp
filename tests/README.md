# NOCP Tests

This directory contains the test suite for the NOCP (High-Efficiency LLM Proxy Agent) project.

## Test Organization

Tests are organized by category for better maintainability and faster execution:

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and pytest configuration
├── run_tests.py                   # Custom test runner
│
├── unit/                          # Fast, isolated tests (<100ms each)
│   ├── README.md
│   ├── test_config.py            # Configuration validation
│   ├── test_token_counter.py     # Token counting
│   ├── test_tool_executor.py     # Tool execution (mocked)
│   ├── test_context_manager.py   # Compression logic (mocked LLM)
│   ├── test_output_serializer.py # TOON encoding/decoding
│   ├── test_router.py            # Request routing
│   ├── test_result.py            # Result pattern
│   └── test_llm_client.py        # LLM client (mocked)
│
├── integration/                   # Component integration (may call APIs)
│   ├── README.md
│   ├── test_agent_flow.py        # Act → Assess → Articulate flow
│   ├── test_compression_pipeline.py  # Full compression pipeline
│   ├── test_llm_integration.py   # LiteLLM integration
│   ├── test_serialization_pipeline.py
│   └── test_conversation_history.py
│
├── e2e/                          # End-to-end workflows
│   ├── README.md
│   ├── test_complete_workflow.py # Full request → response
│   ├── test_gemini_integration.py # Real Gemini API calls
│   └── test_multi_turn.py        # Multi-turn conversations
│
└── performance/                  # Performance benchmarks
    ├── README.md
    ├── test_compression_speed.py
    ├── test_throughput.py
    └── test_memory_usage.py
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run only fast tests (default, skips slow tests)
pytest tests/

# Run specific category
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/           # E2E tests
pytest tests/performance/   # Performance tests
```

### Using the Custom Test Runner

```bash
# Run unit tests only
python tests/run_tests.py unit

# Run integration tests
python tests/run_tests.py integration

# Run e2e tests (includes slow tests)
python tests/run_tests.py e2e

# Run performance benchmarks
python tests/run_tests.py performance

# Run only fast tests
python tests/run_tests.py fast

# Run all tests including slow ones
python tests/run_tests.py all
```

### Using Markers

```bash
# Run tests by marker
pytest -m unit              # Unit tests
pytest -m integration       # Integration tests
pytest -m e2e               # E2E tests
pytest -m performance       # Performance tests

# Run slow tests
pytest --run-slow

# Run tests requiring API key
pytest -m requires_api_key
```

## Test Categories

### Unit Tests (`tests/unit/`)

- **Purpose**: Fast, isolated tests with no external dependencies
- **Speed**: <100ms per test
- **Mocking**: All external dependencies mocked
- **When to run**: Frequently during development

### Integration Tests (`tests/integration/`)

- **Purpose**: Test component interactions
- **Speed**: May take longer (100ms - 1s)
- **Mocking**: May use mocked or real APIs
- **When to run**: Before commits, in CI

### E2E Tests (`tests/e2e/`)

- **Purpose**: Full workflow validation
- **Speed**: Slower (>1 second)
- **Mocking**: Minimal, uses real APIs when available
- **When to run**: Before releases, nightly builds

### Performance Tests (`tests/performance/`)

- **Purpose**: Performance benchmarks
- **Speed**: Variable, usually slow
- **Mocking**: Minimal
- **When to run**: For regression testing, optimization validation

## Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.slow` - Tests taking >1 second
- `@pytest.mark.requires_api_key` - Requires GEMINI_API_KEY

## Environment Variables

Some tests require environment variables:

- `GEMINI_API_KEY` - For Gemini API integration tests

## CI/CD Integration

The test suite is designed for efficient CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Unit Tests
  run: pytest tests/unit/

- name: Integration Tests
  run: pytest tests/integration/

- name: E2E Tests (on main only)
  if: github.ref == 'refs/heads/main'
  run: pytest tests/e2e/ --run-slow
```

## Coverage

Coverage is optional and not enabled by default for faster development iterations.

Generate coverage reports when needed:

```bash
# Run with coverage (terminal + HTML report)
pytest --cov=nocp --cov-report=term-missing --cov-report=html

# Run specific category with coverage
pytest tests/unit/ --cov=nocp --cov-report=html

# View HTML coverage report
open htmlcov/index.html
```

**Note**: Coverage is recommended for CI/CD pipelines and before releases, but not required for quick development test runs.

## Writing Tests

### Test Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Test Organization

1. Group related tests in classes
2. Use descriptive test names
3. One assertion concept per test
4. Use fixtures for common setup

### Example

```python
import pytest


@pytest.mark.unit
class TestToolExecutor:
    """Tests for ToolExecutor."""

    def test_successful_execution(self, tool_executor):
        """Test successful tool execution."""
        # Arrange
        request = ToolRequest(...)

        # Act
        result = tool_executor.execute(request)

        # Assert
        assert result.success is True
```

## Troubleshooting

### Tests Not Found

Ensure you're running pytest from the project root:

```bash
cd /path/to/nocp
pytest tests/
```

### Import Errors

Install the package in development mode:

```bash
pip install -e ".[dev]"
```

### Slow Tests Running

By default, slow tests are skipped. Enable them with:

```bash
pytest --run-slow
```

## Contributing

When adding new tests:

1. Place in appropriate category directory
2. Add appropriate markers
3. Update category README if needed
4. Ensure tests are fast (<100ms for unit tests)
5. Mock external dependencies in unit tests
