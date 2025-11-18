# NOCP - High-Efficiency LLM Proxy Agent

**Token-Oriented Optimization Layer for Large Context Models**

A Python-based LLM orchestration middleware that optimizes token usage for ultra-large context models (like Gemini 2.5 Flash) through dynamic compression and intelligent serialization, targeting 50-70% input token reduction and 30-60% output token reduction.

## Architecture Overview

NOCP implements the **Act-Assess-Articulate** pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    High-Efficiency Proxy Agent                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────┐   │
│  │  Request   │───▶│    Tool    │───▶│     Context        │   │
│  │  Router    │    │  Executor  │    │     Manager        │   │
│  └────────────┘    │   (Act)    │    │    (Assess)        │   │
│                    └────────────┘    └────────────────────┘   │
│                           │                      │             │
│                           ▼                      ▼             │
│                    ┌────────────────────────────────┐          │
│                    │      Gemini 2.5 Flash LLM     │          │
│                    └────────────────────────────────┘          │
│                           │                                    │
│                           ▼                                    │
│                    ┌────────────┐                              │
│                    │   Output   │                              │
│                    │ Serializer │                              │
│                    │(Articulate)│                              │
│                    └────────────┘                              │
│                           │                                    │
└───────────────────────────┼────────────────────────────────────┘
                            ▼
                     TOON/JSON Output
```

### Core Components

1. **Request Router**: Parses queries, validates schemas, prepares minimal context
2. **Tool Executor (Act)**: Executes external functions with Pydantic validation
3. **Context Manager (Assess)**: Applies dynamic compression via:
   - Semantic Pruning (RAG/document outputs)
   - Knowledge Distillation (Student Summarizer model)
   - Conversation History Compaction
4. **Output Serializer (Articulate)**: Converts responses to TOON format

## Key Features

### Dynamic Context Compression

- **Token Gate**: Uses Gemini's `CountTokens` API to measure before compression
- **Cost-of-Compression Calculus**: Only compresses if net savings justify overhead
- **Adaptive Thresholds**: Tool-specific compression thresholds (T_comp)
- **Target**: 50-70% reduction in tool output tokens

### TOON Serialization

- **Format Negotiation**: Automatically selects TOON vs compact JSON based on data structure
- **Tabularity Analysis**: Evaluates data for optimal serialization format
- **Target**: 30-60% reduction vs formatted JSON

### Monitoring & Drift Detection

- **Context Watchdog**: Tracks efficiency delta across transactions
- **Comprehensive Metrics**: Per-transaction token counts, costs, latencies
- **Drift Alerts**: Warns when compression ratios degrade over time

## Installation

```bash
# Clone the repository
git clone https://github.com/Raudbjorn/nocp.git
cd nocp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For LiteLLM multi-cloud support
pip install -e ".[litellm]"
```

## Configuration

Create a `.env` file in the project root:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Compression settings
DEFAULT_COMPRESSION_THRESHOLD=5000
ENABLE_SEMANTIC_PRUNING=true
ENABLE_KNOWLEDGE_DISTILLATION=true
ENABLE_HISTORY_COMPACTION=true

# Output settings
DEFAULT_OUTPUT_FORMAT=toon
ENABLE_FORMAT_NEGOTIATION=true

# Monitoring
ENABLE_METRICS_LOGGING=true
METRICS_LOG_FILE=./logs/metrics.jsonl
```

## Quick Start

```python
from nocp import HighEfficiencyProxyAgent, AgentRequest
from nocp.models.schemas import ToolDefinition, ToolParameter

# Initialize agent
agent = HighEfficiencyProxyAgent()

# Define a tool
def search_products(query: str) -> str:
    # Your tool implementation
    return "Product results..."

tool_def = ToolDefinition(
    name="search_products",
    description="Search product catalog",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True
        )
    ]
)

# Register tool
agent.register_tool(tool_def, search_products)

# Process request
request = AgentRequest(
    query="Find wireless headphones under $100",
    session_id="user-123"
)

response, metrics = agent.process_request(request)

print(f"Response: {response}")
print(f"Token Savings: {metrics.raw_input_tokens - metrics.compressed_input_tokens}")
print(f"Cost: ${metrics.estimated_cost_usd:.6f}")
```

## Running Examples

```bash
# Basic demo
python examples/demo_basic.py

# Ensure GEMINI_API_KEY is set
export GEMINI_API_KEY='your-key-here'
python examples/demo_basic.py
```

## Architecture Deep Dive

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architectural documentation.

### Token Economy

The system optimizes two distinct phases:

| Phase | Strategy | Target Reduction | Method |
|-------|----------|------------------|--------|
| **Input Context** | Dynamic Compression | 50-70% | Semantic Pruning, Knowledge Distillation |
| **Output Response** | TOON Serialization | 30-60% | Format Negotiation + TOON encoding |

## Development

### Project Structure

```
nocp/
├── src/nocp/
│   ├── core/           # Core agent and configuration
│   ├── models/         # Pydantic schemas
│   ├── modules/        # Act-Assess-Articulate components
│   ├── utils/          # Logging, token counting
│   └── tools/          # Example tools
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── examples/           # Demo scripts
├── docs/              # Documentation
└── config/            # Configuration files
```

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=nocp --cov-report=html

# Specific test
pytest tests/unit/test_context_manager.py
```

### Code Quality

```bash
# Format code
black src/nocp

# Lint
ruff check src/nocp

# Type checking
mypy src/nocp
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Input token reduction | 50-70% | ✅ Implemented |
| Output token reduction | 30-60% | ✅ Implemented |
| Cost-of-Compression validation | 100% | ✅ Implemented |
| Latency overhead | <15% | ⚠️  Needs optimization |
| Drift detection | Real-time | ✅ Implemented |

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: [docs/](docs/)
- Issues: GitHub Issues
- Architectural Blueprint: See project documentation

---

**Built with focus on token efficiency, cost optimization, and production reliability.**