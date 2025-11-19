# Integration Tests

Component integration tests that may call external APIs (mocked or real).

## Purpose

Integration tests validate how components work together. These tests may:

- Use mocked external APIs (LiteLLM, Gemini)
- Test data flow between components
- Validate error handling across module boundaries
- Test configuration integration

## Coverage

- `test_agent_flow.py` - Act → Assess → Articulate flow
- `test_compression_pipeline.py` - Full compression pipeline
- `test_llm_integration.py` - LiteLLM integration (mocked)
- `test_serialization_pipeline.py` - Serialization pipeline
- `test_conversation_history.py` - Conversation history and persistence

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/

# Run with markers
pytest -m integration

# Run specific test file
pytest tests/integration/test_agent_flow.py
```

## Markers

- `@pytest.mark.integration` - Marks test as integration test
- `@pytest.mark.slow` - For tests taking >1 second (run with --run-slow)

## Notes

- Integration tests may take longer than unit tests
- Some tests use mocked APIs to avoid actual API calls
- Use fixtures to set up complex test scenarios
