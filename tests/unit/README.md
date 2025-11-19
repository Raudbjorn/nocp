# Unit Tests

Fast, isolated tests with no external dependencies (<100ms each).

## Purpose

Unit tests validate individual components in isolation using mocks and stubs for external dependencies. These tests should be:

- **Fast**: Each test completes in <100ms
- **Isolated**: No network calls, file I/O, or database access
- **Deterministic**: Same input always produces same output
- **Focused**: Test one thing at a time

## Coverage

- `test_config.py` - Configuration validation
- `test_token_counter.py` - Token counting logic
- `test_tool_executor.py` - Tool execution (mocked)
- `test_context_manager.py` - Compression logic (mocked LLM)
- `test_output_serializer.py` - TOON encoding/decoding
- `test_router.py` - Request routing
- `test_result.py` - Result pattern
- `test_llm_client.py` - LLM client (mocked APIs)

## Running Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run with verbose output
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_tool_executor.py

# Run only fast tests (default, skips slow tests)
pytest tests/unit/
```

## Markers

- `@pytest.mark.unit` - Marks test as unit test
