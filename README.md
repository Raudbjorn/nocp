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

## Unified Build System (Recommended)

NOCP includes a comprehensive, intelligent build script (`build.sh`) that automatically detects package managers (uv, poetry, pip) and provides a unified interface for all development operations with git/GitHub status monitoring.

### Quick Start

```bash
# One-time setup (auto-detects uv, poetry, or pip)
./build.sh setup

# Build and validate
./build.sh build

# Run tests with coverage
./build.sh test

# Complete QA pipeline
./build.sh format && ./build.sh lint && ./build.sh test
```

### Available Commands

**Setup**
```bash
./build.sh setup              # Install dependencies (auto-detects package manager)
./build.sh setup-litellm      # Install with optional LiteLLM support
```

**Build & Quality**
```bash
./build.sh build              # Build and validate (type checks + linting)
./build.sh lint               # Run linting (ruff + mypy)
./build.sh format             # Format code (ruff format)
```

**Testing**
```bash
./build.sh test               # Run all tests with coverage
./build.sh test-unit          # Run unit tests only
./build.sh test-integration   # Run integration tests only
./build.sh test-e2e           # Run end-to-end tests only
./build.sh benchmark          # Run performance benchmarks
```

**Development**
```bash
./build.sh example basic_usage    # Run an example
./build.sh examples               # List available examples
./build.sh docs                   # Generate API documentation
```

**Utilities**
```bash
./build.sh status             # Detailed git/GitHub repository status
./build.sh clean              # Remove all build artifacts
./build.sh clean-cache        # Remove only cache directories
./build.sh help               # Show full help
```

### Git & GitHub Integration

The build script provides intelligent repository awareness:

- **Uncommitted Changes**: Alerts when >5 files, warns at >20
- **Branch Divergence**: Shows commits behind/ahead of main/master
- **Merge Conflicts**: Detects potential conflicts before merging
- **Pull Requests**: Shows open PRs, draft status, and merge conflicts (via `gh` CLI)
- **CI/CD Status**: Displays recent workflow failures
- **Review Requests**: Shows PRs awaiting your review

Example status output:
```bash
./build.sh status
# ⚠️ Git Status Notifications:
#   📝 You have 8 uncommitted changes
#   📤 You have 2 unpushed commits on branch 'feature/new-compression'
#   🔀 There are 3 open pull request(s) in Raudbjorn/nocp
#     • #4: Add multi-cloud routing (@svnbjrn)
#     • #5: Comprehensive test coverage (@svnbjrn)
```

### Package Manager Detection

The build script automatically detects and uses the best available Python package manager:

1. **uv** (fastest, recommended) - Prioritized if available
2. **poetry** - Used if uv not found
3. **pip + venv** - Fallback for standard Python installations

No configuration needed - it just works!

### Environment Variables

```bash
# Install with LiteLLM
INSTALL_LITELLM=true ./build.sh setup

# Set API key for examples
export GEMINI_API_KEY='your-key-here'
./build.sh example basic_usage
```

## Configuration

NOCP supports multiple configuration methods with the following precedence (highest to lowest):

1. **CLI arguments** - Explicit kwargs passed to `ProxyConfig()`
2. **Environment variables** - Prefixed with `NOCP_`
3. **.env file** - Local environment configuration
4. **pyproject.toml** - Project-specific defaults via `[tool.nocp]` section
5. **Hardcoded defaults** - Built-in fallback values

### Option 1: Environment Variables (.env file)

Create a `.env` file in the project root:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings
NOCP_GEMINI_API_KEY=your_gemini_api_key_here
NOCP_GEMINI_MODEL=gemini-2.5-flash

# Compression settings
NOCP_DEFAULT_COMPRESSION_THRESHOLD=5000
NOCP_ENABLE_SEMANTIC_PRUNING=true
NOCP_ENABLE_KNOWLEDGE_DISTILLATION=true
NOCP_ENABLE_HISTORY_COMPACTION=true

# Output settings
NOCP_DEFAULT_OUTPUT_FORMAT=toon
NOCP_ENABLE_FORMAT_NEGOTIATION=true

# Monitoring
NOCP_ENABLE_METRICS_LOGGING=true
NOCP_METRICS_LOG_FILE=./logs/metrics.jsonl
```

### Option 2: PyProject.toml (Recommended for Teams)

Add project-specific defaults to your `pyproject.toml`:

```toml
[tool.nocp]
# Project-specific NOCP defaults
default_compression_threshold = 5000
enable_semantic_pruning = true
enable_knowledge_distillation = false
default_output_format = "toon"
log_level = "INFO"

# LLM settings
litellm_default_model = "gemini/gemini-2.0-flash-exp"
litellm_fallback_models = "gemini/gemini-1.5-flash,openai/gpt-4o-mini"

# Metrics
enable_metrics_logging = true
metrics_log_file = ".nocp/metrics.jsonl"
```

**Benefits of pyproject.toml configuration:**
- ✅ Share configuration via version control
- ✅ No need to set environment variables for project defaults
- ✅ Follows Python packaging standards (like `tool.ruff`, `tool.mypy`)
- ✅ Clear separation: `pyproject.toml` (project) vs `.env` (local secrets)

**Note:** Keep API keys in `.env` (not in version control), use `pyproject.toml` for project defaults.

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

### Pre-commit Hooks (Recommended)

NOCP uses pre-commit hooks to automatically enforce code quality standards before each commit. This ensures consistent formatting, catches common issues early, and reduces CI failures.

#### Setup

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

#### What the hooks do

The pre-commit configuration automatically runs the following checks before each commit:

- **Code Formatting**: Black automatically formats Python code to ensure consistency
- **Linting**: Ruff checks for code quality issues and auto-fixes where possible
- **Type Checking**: MyPy validates type annotations
- **General Checks**:
  - Remove trailing whitespace
  - Ensure files end with a newline
  - Validate YAML, JSON, and TOML syntax
  - Check for large files (>1MB)
  - Detect debug statements
  - Verify docstrings come first

#### Manual execution

```bash
# Run hooks on all files
pre-commit run --all-files

# Run hooks on staged files only
pre-commit run

# Update hook versions
pre-commit autoupdate
```

#### Benefits

- Automatic code formatting before commit
- Catch issues early (before CI)
- Consistent code style across contributors
- Reduced CI failures and faster reviews

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
