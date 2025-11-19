# End-to-End Tests

Full workflow tests from request to response.

## Purpose

E2E tests validate complete user workflows through the entire system. These tests:

- Test real API integrations (when API keys available)
- Validate complete request → response flows
- Test multi-turn conversations
- Ensure system works as a cohesive whole

## Coverage

- `test_complete_workflow.py` - Full request → response flow
- `test_gemini_integration.py` - Real Gemini API calls
- `test_multi_turn.py` - Multi-turn conversations

## Running E2E Tests

```bash
# Run all e2e tests
pytest tests/e2e/

# Run with markers
pytest -m e2e

# Run with slow tests enabled
pytest tests/e2e/ --run-slow

# Run tests requiring API key
pytest -m requires_api_key
```

## Environment Variables

Some E2E tests require:

- `GEMINI_API_KEY` - For Gemini integration tests

## Markers

- `@pytest.mark.e2e` - Marks test as end-to-end test
- `@pytest.mark.slow` - For tests taking >1 second
- `@pytest.mark.requires_api_key` - Requires GEMINI_API_KEY

## Notes

- E2E tests may incur API costs
- These tests are slower and should run less frequently
- Skip tests requiring API keys if not configured
