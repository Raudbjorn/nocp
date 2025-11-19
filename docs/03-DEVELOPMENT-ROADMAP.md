# Development Roadmap

## Overview

This document outlines the phased development plan for the LLM Proxy Agent, with clear milestones, deliverables, and success criteria.

---

## Phase 0: Bootstrap Infrastructure (Week 1, Days 1-2)

**Goal**: Establish project foundation with uv-based tooling that is transparent to users.

### Deliverables

#### D0.1: Project Structure
```
nocp/
├── pyproject.toml          # uv-compatible project metadata
├── README.md               # User-facing documentation
├── nocp                    # Single executable entry point (shell script)
├── src/
│   └── nocp/
│       ├── __init__.py
│       ├── __main__.py     # Python entry point
│       ├── bootstrap.py    # uv auto-installer
│       └── cli.py          # Command-line interface
├── tests/
│   ├── __init__.py
│   └── conftest.py         # pytest configuration
├── docs/                   # Specification documents (completed)
└── .gitignore
```

#### D0.2: uv Bootstrap Script (`src/nocp/bootstrap.py`)

**Requirements**:
- Detect if uv is installed
- Auto-install uv if missing (platform-specific)
- Transparent to user (no manual intervention)
- Fallback to system Python if uv unavailable

**Success Criteria**:
- ✅ `./nocp --version` works on clean system
- ✅ uv installed to user directory (no sudo required)
- ✅ Cross-platform support (Linux, macOS, Windows via WSL)

#### D0.3: Single Executable Entry Point (`nocp`)

Shell script that:
1. Sources bootstrap.py to ensure uv available
2. Proxies all commands to `uv run python -m nocp`
3. Provides seamless experience

```bash
#!/usr/bin/env bash
# ./nocp entry point

# Bootstrap uv if needed
python3 src/nocp/bootstrap.py || exit 1

# Proxy to uv
exec uv run python -m nocp "$@"
```

**Commands to support**:
- `./nocp --help`: Show CLI help
- `./nocp --version`: Show version info
- `./nocp setup`: Initialize dependencies
- `./nocp run <script>`: Execute Python script with project deps
- `./nocp test`: Run test suite
- `./nocp benchmark`: Run benchmarks

#### D0.4: Dependency Management (`pyproject.toml`)

```toml
[project]
name = "nocp"
version = "0.1.0"
description = "High-efficiency LLM Proxy Agent with token optimization"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.9.0",
    "litellm>=1.55.0",
    "rich>=13.0.0",      # CLI formatting
    "typer>=0.15.0",     # CLI framework
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
```

### Timeline
- **Day 1**: Project structure, bootstrap script, entry point
- **Day 2**: CLI framework, basic commands, documentation

### Acceptance Criteria
- [ ] `./nocp setup` installs all dependencies via uv
- [ ] `./nocp test` runs pytest suite
- [ ] `./nocp --version` shows correct version
- [ ] No manual uv installation required
- [ ] Works on fresh Ubuntu/macOS system

---

## Phase 1: Core Modules (Week 1-2, Days 3-10)

**Goal**: Implement Act, Assess, and Articulate modules with basic functionality.

### Milestone 1.1: Act Module - Tool Executor (Days 3-4)

#### Deliverables

**D1.1.1: Base Tool Executor**
- File: `src/nocp/core/act.py`
- Implements `ToolExecutor` protocol
- Supports: Python functions, mock database queries
- Includes retry logic and timeout handling

**D1.1.2: Tool Registry**
- Dynamic tool registration system
- Validates tool signatures
- Provides tool discovery

**D1.1.3: Unit Tests**
- File: `tests/core/test_act.py`
- Coverage: >90%
- Tests: successful execution, errors, retries, timeouts

**Code Example**:
```python
# Usage
from nocp.core.act import ToolExecutor, ToolRequest

executor = ToolExecutor()

# Register a tool
@executor.register_tool("fetch_user_data")
def fetch_user_data(user_id: str) -> dict:
    return {"user_id": user_id, "name": "John Doe"}

# Execute
request = ToolRequest(
    tool_id="fetch_user_data",
    tool_type="python_function",
    function_name="fetch_user_data",
    parameters={"user_id": "123"}
)
result = executor.execute(request)
assert result.success is True
```

#### Success Criteria
- [ ] Execute Python functions with parameter validation
- [ ] Handle errors gracefully with retry logic
- [ ] Return ToolResult with accurate token estimates
- [ ] All tests passing

---

### Milestone 1.2: Assess Module - Context Manager (Days 5-7)

#### Deliverables

**D1.2.1: Token Counter**
- Integration with LiteLLM's token counting
- Support for multiple model tokenizers
- Accurate estimates (±5% of actual)

**D1.2.2: Semantic Pruning (RAG Output)**
- Implement top-k chunk selection
- Embedding-based similarity filtering
- Target: 60-70% reduction for document-heavy inputs

**D1.2.3: Basic Summarization (Student Model)**
- Integration with lightweight LLM (gpt-4o-mini)
- Cost-benefit calculation (only compress if savings > overhead)
- Fallback to raw output if compression too expensive

**D1.2.4: Unit Tests**
- File: `tests/core/test_assess.py`
- Tests: token counting accuracy, compression ratios, fallback logic

**Code Example**:
```python
from nocp.core.assess import ContextManager, ContextData

manager = ContextManager(
    student_model="openai/gpt-4o-mini",
    compression_threshold=10_000
)

context = ContextData(
    tool_results=[large_tool_result],
    transient_context={"query": "What are the top insights?"},
    max_tokens=50_000
)

optimized = manager.optimize(context)
assert optimized.compression_ratio < 0.5  # >50% reduction
assert optimized.optimized_tokens < context.max_tokens
```

#### Success Criteria
- [ ] Token counting within ±5% accuracy
- [ ] Semantic pruning achieves >60% reduction on test dataset
- [ ] Summarization cost < savings (validated on sample)
- [ ] All tests passing

---

### Milestone 1.3: Articulate Module - Output Serializer (Days 8-10)

#### Deliverables

**D1.3.1: TOON Serializer**
- Implement TOON encoding/decoding
- Support for nested objects and arrays
- Length markers for validation

**D1.3.2: Format Negotiation Layer**
- Analyze Pydantic schema complexity
- Detect tabular vs nested structures
- Auto-select TOON or compact JSON

**D1.3.3: Validation Pipeline**
- Round-trip serialization/deserialization tests
- Ensure lossless encoding

**D1.3.4: Unit Tests**
- File: `tests/core/test_articulate.py`
- Tests: TOON encoding, JSON fallback, format negotiation

**Code Example**:
```python
from nocp.core.articulate import OutputSerializer, SerializationRequest
from pydantic import BaseModel

class UserList(BaseModel):
    users: list[dict[str, str]]

data = UserList(users=[
    {"id": "1", "name": "Alice"},
    {"id": "2", "name": "Bob"}
])

serializer = OutputSerializer()
request = SerializationRequest(data=data)
output = serializer.serialize(request)

assert output.format_used == "toon"  # Tabular data
assert output.savings_ratio > 0.3  # >30% savings
assert output.is_valid is True
```

#### Success Criteria
- [ ] TOON encoding achieves >30% savings on tabular data
- [ ] Format negotiation selects correct format (90% accuracy on test suite)
- [ ] Round-trip validation passes for all test cases
- [ ] All tests passing

---

## Phase 2: Integration and Orchestration (Week 2-3, Days 11-15)

**Goal**: Connect all modules into end-to-end pipeline with LLM integration.

### Milestone 2.1: LiteLLM Integration (Days 11-12)

#### Deliverables

**D2.1.1: LLM Client Wrapper**
- File: `src/nocp/llm/client.py`
- Unified interface to LiteLLM
- Support for structured output (JSON schema)
- Error handling and retries

**D2.1.2: Model Router**
- Cost-based routing logic
- Complexity scoring heuristic
- Configuration for model tiers

**Code Example**:
```python
from nocp.llm.client import LLMClient

client = LLMClient(default_model="gemini/gemini-2.0-flash-exp")

response = client.complete(
    messages=[{"role": "user", "content": optimized_context}],
    response_schema=OutputSchema,  # Pydantic model
    max_tokens=10_000
)

assert isinstance(response.content, OutputSchema)
```

---

### Milestone 2.2: Orchestrator (HighEfficiencyProxyAgent) (Days 13-15)

#### Deliverables

**D2.2.1: Main Orchestrator Class**
- File: `src/nocp/agent.py`
- Implements full Act → Assess → LLM → Articulate pipeline
- Handles errors with graceful degradation
- Collects metrics at each stage

**D2.2.2: End-to-End Tests**
- File: `tests/test_e2e.py`
- Tests: full pipeline execution, error handling, metrics collection

**D2.2.3: Example Use Case**
- File: `examples/basic_usage.py`
- Demonstrates complete workflow

**Code Example**:
```python
from nocp.agent import HighEfficiencyProxyAgent, ProxyRequest

agent = HighEfficiencyProxyAgent()

request = ProxyRequest(
    query="Summarize the top 5 customer complaints",
    required_tools=[
        ToolRequest(tool_id="db_query", function_name="fetch_complaints", ...)
    ],
    enable_compression=True,
    enable_toon=True
)

response = agent.process(request)

print(f"Cost savings: ${response.cost_analysis.savings:.4f}")
print(f"Latency: {response.total_latency_ms:.0f}ms")
print(f"Result: {response.result.serialized_text}")
```

#### Success Criteria
- [ ] End-to-end pipeline executes successfully
- [ ] All modules integrated correctly
- [ ] Metrics collected at each stage
- [ ] Example runs without errors

---

## Phase 3: Optimization and Monitoring (Week 3-4, Days 16-21)

**Goal**: Add advanced features, monitoring, and benchmarking.

### Milestone 3.1: Conversation History Compaction (Days 16-17)

#### Deliverables
- Implement roll-up summarization for chat history
- Maintain conversation state across sessions
- Tests for multi-turn conversations

---

### Milestone 3.2: Observability Infrastructure (Days 18-19)

#### Deliverables

**D3.2.1: Structured Logging**
- File: `src/nocp/observability/logging.py`
- JSON-structured logs
- Log all transactions with `TransactionLog` schema

**D3.2.2: Metrics Collection**
- Track compression ratios, latency, costs
- Export to console/file

**D3.2.3: Drift Detection**
- Monitor efficiency delta over rolling window
- Alert on degradation

---

### Milestone 3.3: Benchmarking Suite (Days 20-21)

#### Deliverables

**D3.3.1: Benchmark Framework**
- File: `benchmarks/run_benchmarks.py`
- Compare optimized vs baseline pipeline
- Generate reports with charts

**D3.3.2: Test Datasets**
- Synthetic data for RAG, API calls, database queries
- Varying sizes (small, medium, large)

**D3.3.3: Performance Reports**
- Markdown reports with tables and charts
- Track KPIs: token reduction, cost savings, latency

#### Success Criteria
- [ ] Benchmarks show >50% input token reduction
- [ ] Benchmarks show >30% output token reduction
- [ ] End-to-end latency <2x baseline
- [ ] Cost reduction >40%

---

## Phase 4: Production Readiness (Week 4-5, Days 22-28)

### Milestone 4.1: Caching Layer (Days 22-23)
- In-memory LRU cache for tool results
- Optional ChromaDB integration for distributed caching

### Milestone 4.2: Async Support (Days 24-25)
- Async versions of all modules
- Concurrent tool execution
- Performance benchmarks for async vs sync

### Milestone 4.3: Documentation and Examples (Days 26-28)
- Complete API reference
- User guide with tutorials
- Advanced examples (RAG, multi-turn chat, API aggregation)

---

## Release Checklist

### MVP Release (v0.1.0)
- [ ] All Phase 1-2 milestones complete
- [ ] Unit test coverage >85%
- [ ] End-to-end tests passing
- [ ] Basic benchmarks run successfully
- [ ] README with quickstart guide
- [ ] uv bootstrap working on all platforms

### Beta Release (v0.2.0)
- [ ] All Phase 3 milestones complete
- [ ] Observability infrastructure operational
- [ ] Benchmarking suite with reports
- [ ] Documentation complete

### Production Release (v1.0.0)
- [ ] All Phase 4 milestones complete
- [ ] Async support fully tested
- [ ] Caching layer operational
- [ ] Security audit complete
- [ ] Performance meets all KPIs

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TOON library unavailable | Medium | High | Implement TOON encoder from spec |
| LiteLLM API changes | Low | Medium | Pin versions, monitor changelog |
| Compression overhead > savings | Medium | High | Cost-benefit checks, fallback to raw |
| uv installation fails | Low | High | Fallback to pip, clear error messages |
| Token counting inaccurate | Medium | Medium | Validate against actual usage, adjust |

---

**Next**: See `04-COMPONENT-SPECS.md` for detailed implementation specifications.
