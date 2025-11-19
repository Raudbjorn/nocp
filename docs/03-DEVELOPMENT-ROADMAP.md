# Development Roadmap

## Overview

This document outlines the phased development plan for the LLM Proxy Agent (NOCP), with clear milestones, deliverables, and success criteria.

**Latest Update**: Added Phase 6 (MDMAI-Inspired Enterprise Enhancements) based on comprehensive analysis of the MDMAI project, bringing enterprise-grade patterns for multi-provider orchestration, advanced caching, and production security.

---

## Table of Contents

1. [Phase 0: Bootstrap Infrastructure](#phase-0-bootstrap-infrastructure-week-1-days-1-2)
2. [Phase 1: Core Modules](#phase-1-core-modules-week-1-2-days-3-10)
3. [Phase 2: Integration and Orchestration](#phase-2-integration-and-orchestration-week-2-3-days-11-15)
4. [Phase 3: Optimization and Monitoring](#phase-3-optimization-and-monitoring-week-3-4-days-16-21)
5. [Phase 4: Production Readiness](#phase-4-production-readiness-week-4-5-days-22-28)
6. [**Phase 5: Code Quality & Infrastructure**](#phase-5-code-quality--infrastructure-improvements)
7. [**Phase 6: MDMAI-Inspired Enterprise Enhancements (NEW)**](#phase-6-mdmai-inspired-enterprise-enhancements)
8. [Release Checklist](#release-checklist)
9. [Risk Mitigation](#risk-mitigation)

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
- Optional Redis integration for distributed caching

### Milestone 4.2: Async Support (Days 24-25)
- Async versions of all modules
- Concurrent tool execution
- Performance benchmarks for async vs sync

### Milestone 4.3: Documentation and Examples (Days 26-28)
- Complete API reference
- User guide with tutorials
- Advanced examples (RAG, multi-turn chat, API aggregation)

---

## Phase 5: Code Quality & Infrastructure Improvements

**Status**: 🆕 NEW - Based on comprehensive analysis of mature rapydocs codebase
**Goal**: Elevate NOCP to production-grade quality with mature development infrastructure, enhanced developer experience, and industry-standard tooling.

**Context**: After analyzing the mature rapydocs project (470+ lines of config, comprehensive testing infrastructure, CI/CD), we identified 42 concrete improvements across 10 categories that would significantly enhance NOCP's quality, maintainability, and developer experience.

### Overview

This phase focuses on **non-functional requirements** that transform NOCP from a working prototype into a production-ready, maintainable, and professionally developed project. These improvements don't add new features but make the codebase more robust, easier to maintain, and more pleasant to work with.

**Total Estimated Effort**: 48-62 hours across 4 sub-phases

---

### Phase 5.1: Enhanced Configuration & Validation (Priority: HIGH)

**Effort**: 8-10 hours | **Impact**: High - Better type safety, validation, flexibility

#### Background

The rapydocs project demonstrates sophisticated configuration management with 470 lines in `config.py` featuring:
- Enums for all configuration choices (prevents typos, enables IDE autocomplete)
- Pydantic field validators catching errors at startup
- Multi-source configuration precedence: CLI > env > .env > pyproject.toml
- Configuration export/import for sharing and debugging
- Hardware-specific auto-tuning

**NOCP Current State**: 207 lines, basic pydantic-settings, limited validation, no enums.

#### Deliverables

**D5.1.1: Configuration Enums** (2 hours)

Create enums for all configuration choices to improve type safety and developer experience.

**File**: `src/nocp/models/enums.py`

```python
"""Configuration enums for type-safe settings"""
from enum import Enum

class OutputFormat(str, Enum):
    """Supported output formats"""
    TOON = "toon"
    COMPACT_JSON = "compact_json"
    JSON = "json"

    def __str__(self) -> str:
        return self.value

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class CompressionStrategy(str, Enum):
    """Available compression strategies"""
    SEMANTIC_PRUNING = "semantic_pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    HISTORY_COMPACTION = "history_compaction"
    NONE = "none"

class LLMProvider(str, Enum):
    """Supported LLM providers via LiteLLM"""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    AZURE = "azure"
```

**Usage in ProxyConfig**:

```python
from .models.enums import OutputFormat, LogLevel, CompressionStrategy

class ProxyConfig(BaseSettings):
    default_output_format: OutputFormat = Field(
        default=OutputFormat.TOON,
        description="Default serialization format"
    )

    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Application log level"
    )

    compression_strategies: List[CompressionStrategy] = Field(
        default=[
            CompressionStrategy.SEMANTIC_PRUNING,
            CompressionStrategy.KNOWLEDGE_DISTILLATION
        ],
        description="Enabled compression strategies"
    )
```

**Benefits**:
- ✅ IDE autocomplete for all config values
- ✅ Runtime validation prevents typos (no more `"ton"` vs `"toon"`)
- ✅ Self-documenting configuration options
- ✅ Type checker catches invalid values

**Acceptance Criteria**:
- [ ] All magic strings replaced with enums
- [ ] ProxyConfig uses enums for all choice fields
- [ ] Documentation updated with enum options
- [ ] Tests validate enum constraints

---

**D5.1.2: PyProject.toml Configuration Defaults** (3 hours)

Enable project-specific configuration via `[tool.nocp]` section in pyproject.toml.

**File**: `src/nocp/core/config.py` (enhancement)

```python
import tomllib
from pathlib import Path
from typing import Dict, Any

def load_pyproject_defaults() -> Dict[str, Any]:
    """
    Load defaults from [tool.nocp] section in pyproject.toml.

    Precedence: CLI args > env vars > .env file > pyproject.toml > hardcoded defaults

    Returns:
        Dictionary of configuration overrides from pyproject.toml
    """
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        return {}

    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)

        # Extract [tool.nocp] section
        tool_config = data.get("tool", {}).get("nocp", {})

        logger.debug(f"Loaded {len(tool_config)} settings from pyproject.toml")
        return tool_config

    except Exception as e:
        logger.warning(f"Could not load pyproject.toml: {e}")
        return {}

class ProxyConfig(BaseSettings):
    """Enhanced configuration with pyproject.toml support"""

    model_config = SettingsConfigDict(
        env_prefix="NOCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        # Load pyproject.toml defaults first
        pyproject_defaults = load_pyproject_defaults()

        # Merge with explicit kwargs (kwargs take precedence)
        merged = {**pyproject_defaults, **kwargs}

        super().__init__(**merged)
```

**Example pyproject.toml**:

```toml
[tool.nocp]
# Project-specific NOCP defaults
compression_threshold = 5000
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

**Benefits**:
- ✅ Team projects can share configuration via version control
- ✅ No need to set environment variables for project-specific settings
- ✅ Follows Python packaging standards (similar to tool.ruff, tool.mypy)
- ✅ Clear separation: pyproject.toml (project) vs .env (local secrets)

**Acceptance Criteria**:
- [ ] ProxyConfig loads from pyproject.toml
- [ ] Precedence order maintained: CLI > env > .env > pyproject.toml
- [ ] Documentation explains configuration precedence
- [ ] Tests verify loading from pyproject.toml

---

**D5.1.3: Field Validators** (2 hours)

Add Pydantic field validators to catch configuration errors at startup.

```python
from pydantic import field_validator, model_validator

class ProxyConfig(BaseSettings):
    # ... existing fields ...

    @field_validator('compression_threshold')
    @classmethod
    def validate_compression_threshold(cls, v: int) -> int:
        """Ensure compression threshold is reasonable"""
        if v < 1000:
            raise ValueError(
                f"compression_threshold ({v}) is too low. "
                "Minimum recommended: 1000 tokens"
            )
        if v > 100_000:
            logger.warning(
                f"Very high compression_threshold ({v:,}). "
                "Compression may rarely trigger."
            )
        return v

    @field_validator('target_compression_ratio')
    @classmethod
    def validate_compression_ratio(cls, v: float) -> float:
        """Ensure compression ratio is valid percentage"""
        if not 0.0 < v < 1.0:
            raise ValueError(
                f"target_compression_ratio must be between 0 and 1, got {v}"
            )
        if v > 0.8:
            logger.warning(
                f"High target_compression_ratio ({v:.0%}). "
                "Only minimal compression will occur."
            )
        return v

    @field_validator('toon_fallback_threshold')
    @classmethod
    def validate_toon_threshold(cls, v: float) -> float:
        """Validate TOON fallback threshold"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"toon_fallback_threshold must be 0.0-1.0, got {v}"
            )
        return v

    @model_validator(mode='after')
    def validate_token_limits(self) -> 'ProxyConfig':
        """Cross-field validation of token limits"""
        if self.max_output_tokens > self.max_input_tokens:
            logger.warning(
                f"max_output_tokens ({self.max_output_tokens:,}) > "
                f"max_input_tokens ({self.max_input_tokens:,}). "
                "This may cause issues with some models."
            )

        if self.compression_threshold > self.max_input_tokens:
            raise ValueError(
                f"compression_threshold ({self.compression_threshold:,}) "
                f"exceeds max_input_tokens ({self.max_input_tokens:,})"
            )

        return self
```

**Benefits**:
- ✅ Catch configuration errors immediately at startup
- ✅ Provide helpful error messages with context
- ✅ Prevent runtime failures from bad configuration
- ✅ Warn about potentially problematic settings

**Acceptance Criteria**:
- [ ] All critical fields have validators
- [ ] Validators provide clear error messages
- [ ] Tests verify validation behavior
- [ ] Documentation mentions validation rules

---

**D5.1.4: Configuration Export/Import** (2 hours)

Add ability to save and load configuration for debugging and sharing.

**File**: `src/nocp/utils/config_export.py`

```python
"""Configuration export/import utilities"""
import yaml
from pathlib import Path
from typing import Optional
from ..core.config import ProxyConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

def export_config(
    config: ProxyConfig,
    output_path: Optional[Path] = None,
    include_secrets: bool = False
) -> Path:
    """
    Export configuration to YAML file.

    Args:
        config: Configuration to export
        output_path: Where to save (default: .nocp/config.yaml)
        include_secrets: Whether to include API keys (default: False)

    Returns:
        Path to exported config file
    """
    if output_path is None:
        output_path = Path(".nocp/config.yaml")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export config, excluding secrets by default
    exclude_fields = set()
    if not include_secrets:
        exclude_fields = {
            'gemini_api_key',
            'openai_api_key',
            'anthropic_api_key',
        }

    config_dict = config.model_dump(
        exclude=exclude_fields,
        exclude_none=True,
        mode='json'
    )

    with output_path.open("w") as f:
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=True,
            indent=2
        )

    logger.info(f"Configuration exported to {output_path}")
    return output_path

def import_config(config_path: Path) -> ProxyConfig:
    """
    Import configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        ProxyConfig instance
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config_dict = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return ProxyConfig(**config_dict)

def print_config_diff(config1: ProxyConfig, config2: ProxyConfig) -> None:
    """Print differences between two configurations"""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Configuration Diff")
    table.add_column("Setting", style="cyan")
    table.add_column("Config 1", style="yellow")
    table.add_column("Config 2", style="green")

    dict1 = config1.model_dump(exclude_none=True)
    dict2 = config2.model_dump(exclude_none=True)

    all_keys = sorted(set(dict1.keys()) | set(dict2.keys()))

    for key in all_keys:
        val1 = dict1.get(key, "—")
        val2 = dict2.get(key, "—")

        if val1 != val2:
            table.add_row(key, str(val1), str(val2))

    console.print(table)
```

**CLI commands**:

```bash
# Export current config
./nocp config export --output .nocp/my-config.yaml

# Load config
./nocp config load .nocp/my-config.yaml

# Show current config
./nocp config show

# Compare two configs
./nocp config diff .nocp/config1.yaml .nocp/config2.yaml
```

**Benefits**:
- ✅ Easy configuration debugging
- ✅ Share configurations between team members
- ✅ Configuration presets (dev, staging, prod)
- ✅ Version control configuration without secrets

**Acceptance Criteria**:
- [ ] Export config to YAML (excluding secrets)
- [ ] Import config from YAML
- [ ] CLI commands for export/import/show/diff
- [ ] Tests verify round-trip (export → import)

---

### Phase 5.2: Enhanced Logging & Observability (Priority: HIGH)

**Effort**: 10-14 hours | **Impact**: Very High - Dramatically improves UX and debugging

#### Background

The rapydocs project uses the `rich` library extensively for beautiful terminal output:
- 285 lines of comprehensive logging infrastructure
- Custom logger singleton with rich console
- Progress bars for long operations
- Syntax-highlighted code printing
- Hardware status tables with live updates
- Rich traceback installation for better error messages
- Custom themes for branding

**NOCP Current State**: Basic structlog with JSON/console rendering, no progress indicators, plain console output.

#### Deliverables

**D5.2.1: Rich Console Integration** (4-6 hours)

Integrate `rich` library for dramatically improved terminal output.

**File**: `src/nocp/utils/rich_logging.py`

```python
"""Enhanced logging with Rich library"""
from typing import Optional, Dict, Any
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn
)
from rich.tree import Tree
import structlog

# Custom NOCP theme
NOCP_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "metric": "magenta",
    "token": "blue",
    "cost": "green",
    "savings": "bright_green",
    "latency": "cyan",
    "tool": "blue",
    "compression": "yellow",
})

class NOCPConsole:
    """Singleton console with NOCP branding and theme"""

    _instance: Optional['NOCPConsole'] = None

    def __new__(cls) -> 'NOCPConsole':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.console = Console(theme=NOCP_THEME)
            self.initialized = True

    def print_banner(self):
        """Print NOCP startup banner"""
        self.console.print(
            Panel.fit(
                "[bold cyan]NOCP[/bold cyan] - High-Efficiency LLM Proxy Agent\n"
                "[dim]Token Optimization • Cost Reduction • Smart Compression[/dim]",
                border_style="cyan"
            )
        )

    def print_config_summary(self, config: 'ProxyConfig'):
        """Print configuration summary table"""
        table = Table(title="Configuration", show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Model", config.litellm_default_model)
        table.add_row("Max Input Tokens", f"{config.max_input_tokens:,}")
        table.add_row("Max Output Tokens", f"{config.max_output_tokens:,}")
        table.add_row("Compression Threshold", f"{config.default_compression_threshold:,}")
        table.add_row("Target Compression", f"{config.target_compression_ratio:.0%}")
        table.add_row(
            "Strategies",
            ", ".join([
                "✓ Semantic Pruning" if config.enable_semantic_pruning else "",
                "✓ Knowledge Distillation" if config.enable_knowledge_distillation else "",
                "✓ History Compaction" if config.enable_history_compaction else "",
            ]).strip(", ")
        )

        self.console.print(table)

    def print_metrics(self, metrics: 'ContextMetrics'):
        """Print transaction metrics in beautiful table"""
        table = Table(title="Transaction Metrics", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="yellow", justify="right")
        table.add_column("Details", style="dim")

        # Transaction ID
        table.add_row(
            "Transaction ID",
            metrics.transaction_id[:8] + "...",
            ""
        )

        # Token metrics
        input_savings = metrics.raw_input_tokens - metrics.compressed_input_tokens
        table.add_row(
            "Input Tokens",
            f"{metrics.compressed_input_tokens:,}",
            f"[dim]saved {input_savings:,} ({metrics.input_compression_ratio:.0%})[/dim]"
        )

        output_savings = metrics.raw_output_tokens - metrics.final_output_tokens
        table.add_row(
            "Output Tokens",
            f"{metrics.final_output_tokens:,}",
            f"[dim]saved {output_savings:,} ({metrics.output_compression_ratio:.0%})[/dim]"
        )

        # Total savings
        total_savings = input_savings + output_savings
        table.add_row(
            "[bold]Total Token Savings[/bold]",
            f"[bold green]{total_savings:,}[/bold green]",
            ""
        )

        # Latency breakdown
        table.add_row(
            "Latency",
            f"{metrics.total_latency_ms:.0f}ms",
            f"[dim]compression: {metrics.compression_latency_ms:.0f}ms, "
            f"LLM: {metrics.llm_inference_latency_ms:.0f}ms[/dim]"
        )

        # Format and compression
        table.add_row(
            "Output Format",
            metrics.final_output_format.upper(),
            f"[dim]{len(metrics.tools_used)} tools used[/dim]"
        )

        if metrics.compression_operations:
            ops = ", ".join(metrics.compression_operations)
            table.add_row(
                "Compression",
                f"{len(metrics.compression_operations)} ops",
                f"[dim]{ops}[/dim]"
            )

        self.console.print(table)

    def print_operation_tree(self, operations: Dict[str, Any]):
        """Print operation hierarchy as tree"""
        tree = Tree("[bold cyan]Request Pipeline[/bold cyan]")

        # Act phase
        act_branch = tree.add("[blue]⚡ Act[/blue] - Tool Execution")
        for tool in operations.get('tools', []):
            act_branch.add(f"✓ {tool['name']} ({tool['duration_ms']:.0f}ms)")

        # Assess phase
        assess_branch = tree.add("[yellow]🔍 Assess[/yellow] - Context Optimization")
        for compression in operations.get('compressions', []):
            assess_branch.add(
                f"✓ {compression['method']} "
                f"({compression['ratio']:.0%} reduction)"
            )

        # LLM phase
        llm_branch = tree.add("[magenta]🤖 LLM[/magenta] - Inference")
        llm_branch.add(f"Model: {operations.get('model', 'N/A')}")
        llm_branch.add(f"Tokens: {operations.get('tokens', 0):,}")

        # Articulate phase
        articulate_branch = tree.add("[green]📝 Articulate[/green] - Serialization")
        articulate_branch.add(f"Format: {operations.get('format', 'N/A')}")

        self.console.print(tree)

    def create_progress(self, description: str = "Processing") -> Progress:
        """Create progress bar for long operations"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        )

# Global console instance
console = NOCPConsole()
```

**Usage Examples**:

```python
# In agent.py
from ..utils.rich_logging import console

# Startup
console.print_banner()
console.print_config_summary(self.config)

# Process request with progress
with console.create_progress("Processing request") as progress:
    task = progress.add_task("Executing tools", total=len(tools))

    for tool in tools:
        result = execute_tool(tool)
        progress.advance(task)

# Show results
console.print_metrics(metrics)
console.print_operation_tree(operations)
```

**Benefits**:
- ✅ **Dramatically improved UX** - Professional, modern terminal output
- ✅ **Progress visibility** - Users see what's happening
- ✅ **Better debugging** - Rich tracebacks with syntax highlighting
- ✅ **Metrics visualization** - Beautiful tables instead of JSON logs
- ✅ **Professional appearance** - Makes NOCP feel production-ready

**Acceptance Criteria**:
- [ ] Rich installed and integrated
- [ ] Banner shown on startup
- [ ] Metrics printed in tables
- [ ] Progress bars for long operations
- [ ] Rich tracebacks for errors
- [ ] Custom NOCP theme applied

---

**D5.2.2: Component-Specific Loggers** (2 hours)

Create loggers for each major component with consistent formatting.

**File**: `src/nocp/utils/logging.py` (enhancement)

```python
"""Component-specific logging with rich formatting"""
from typing import Optional
import structlog
from rich.console import Console

class ComponentLogger:
    """Base class for component-specific structured logging"""

    def __init__(self, component_name: str):
        self.logger = structlog.get_logger(component_name)
        self.component = component_name
        self.console = Console()

    def log_operation_start(
        self,
        operation: str,
        details: Optional[dict] = None
    ):
        """Log operation start with emoji"""
        self.console.print(
            f"[cyan]▶[/cyan] [{self.component}] Starting: {operation}"
        )
        if details:
            self.logger.info(
                f"{operation}_started",
                component=self.component,
                **details
            )

    def log_operation_complete(
        self,
        operation: str,
        duration_ms: Optional[float] = None,
        details: Optional[dict] = None
    ):
        """Log operation completion"""
        msg = f"[green]✅[/green] [{self.component}] Completed: {operation}"
        if duration_ms:
            msg += f" ({duration_ms:.0f}ms)"

        self.console.print(msg)

        log_data = {"component": self.component}
        if duration_ms:
            log_data["duration_ms"] = duration_ms
        if details:
            log_data.update(details)

        self.logger.info(f"{operation}_completed", **log_data)

    def log_operation_error(
        self,
        operation: str,
        error: Exception,
        details: Optional[dict] = None
    ):
        """Log operation error"""
        self.console.print(
            f"[red]❌[/red] [{self.component}] Failed: {operation}"
        )
        self.console.print_exception()

        log_data = {
            "component": self.component,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        if details:
            log_data.update(details)

        self.logger.error(f"{operation}_failed", **log_data)

    def log_metric(self, metric_name: str, value: Any, unit: str = ""):
        """Log a metric"""
        self.logger.info(
            "metric",
            component=self.component,
            metric=metric_name,
            value=value,
            unit=unit
        )

# Create component-specific loggers
act_logger = ComponentLogger("act")
assess_logger = ComponentLogger("assess")
articulate_logger = ComponentLogger("articulate")
agent_logger = ComponentLogger("agent")
```

**Usage**:

```python
# In tool_executor.py
from ..utils.logging import act_logger

def execute_tool(self, request: ToolRequest) -> ToolResult:
    act_logger.log_operation_start(
        "tool_execution",
        {"tool_id": request.tool_id}
    )

    try:
        # Execute tool
        result = self._execute(request)

        act_logger.log_operation_complete(
            "tool_execution",
            duration_ms=result.execution_time_ms,
            {"tool_id": request.tool_id, "success": True}
        )

        return result

    except Exception as e:
        act_logger.log_operation_error(
            "tool_execution",
            e,
            {"tool_id": request.tool_id}
        )
        raise
```

**Benefits**:
- ✅ Consistent logging across all components
- ✅ Easy to filter logs by component
- ✅ Rich console output with emoji indicators
- ✅ Structured logs for machine parsing

**Acceptance Criteria**:
- [ ] ComponentLogger class created
- [ ] Loggers created for Act, Assess, Articulate, Agent
- [ ] All components use component loggers
- [ ] Console output includes emoji indicators

---

**D5.2.3: Log File Rotation** (1 hour)

Add rotating file handler to prevent unbounded log growth.

```python
from logging.handlers import RotatingFileHandler
import structlog

def setup_file_logging(
    log_file: Path,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """Configure rotating file handler for logs"""

    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )

    # Configure structlog with file handler
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(file_handler),
    )
```

**Benefits**:
- ✅ Prevents disk space issues
- ✅ Automatic cleanup of old logs
- ✅ Better for production deployments

---

### Phase 5.3: Testing Infrastructure (Priority: HIGH)

**Effort**: 12-15 hours | **Impact**: Very High - Organized testing, better quality

#### Background

The rapydocs project has:
- Custom test runner (278 lines) with category organization
- Test categories: unit, integration, e2e, parsers, database, mcp
- Progress tracking with emoji indicators
- Test timing and performance tracking
- Fail-fast mode and verbose mode
- JSON report generation
- Test summary with success rates

**NOCP Current State**: Basic pytest setup, empty test directories, no custom runner.

#### Deliverables

**D5.3.1: Custom Test Runner** (4 hours)

Create comprehensive test runner with category organization.

**File**: `tests/run_tests.py`

```python
#!/usr/bin/env python3
"""
NOCP Test Runner

Comprehensive test runner with category organization, progress tracking,
and detailed reporting.

Categories:
- unit: Fast, isolated unit tests
- integration: Component integration tests
- e2e: End-to-end workflow tests
- performance: Performance benchmarks

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --category unit    # Run only unit tests
    python tests/run_tests.py -v                 # Verbose output
    python tests/run_tests.py --fail-fast        # Stop on first failure
    python tests/run_tests.py --json report.json # Generate JSON report
"""

import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class TestCategory(Enum):
    """Test categories"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"

@dataclass
class TestResult:
    """Test result for a category"""
    category: str
    passed: bool
    duration_s: float
    test_count: int
    passed_count: int
    failed_count: int
    output: str

@dataclass
class TestSummary:
    """Overall test summary"""
    total_tests: int
    total_passed: int
    total_failed: int
    total_duration_s: float
    success_rate: float
    results: List[TestResult]

class NOCPTestRunner:
    """Test runner for NOCP project"""

    def __init__(
        self,
        verbose: bool = False,
        fail_fast: bool = False,
        capture_output: bool = True
    ):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.capture_output = capture_output
        self.results: List[TestResult] = []

    def discover_tests(
        self,
        category: Optional[TestCategory] = None
    ) -> Dict[TestCategory, List[Path]]:
        """Discover all test files organized by category"""
        categories = {
            TestCategory.UNIT: self.tests_dir / "unit",
            TestCategory.INTEGRATION: self.tests_dir / "integration",
            TestCategory.E2E: self.tests_dir / "e2e",
            TestCategory.PERFORMANCE: self.tests_dir / "performance",
        }

        discovered = {}

        for cat_enum, cat_path in categories.items():
            # Filter by category if specified
            if category and category != cat_enum:
                continue

            # Skip if directory doesn't exist
            if not cat_path.exists():
                continue

            # Find all test_*.py files
            test_files = list(cat_path.rglob("test_*.py"))
            if test_files:
                discovered[cat_enum] = sorted(test_files)

        return discovered

    def run_pytest(
        self,
        test_paths: List[Path],
        category: TestCategory
    ) -> TestResult:
        """Run pytest on given test files"""
        start_time = time.time()

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]

        # Add options
        if self.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        if self.fail_fast:
            cmd.append("-x")

        # Add coverage for unit tests
        if category == TestCategory.UNIT:
            cmd.extend(["--cov=nocp", "--cov-report=term-missing"])

        # Add test paths
        cmd.extend([str(p) for p in test_paths])

        # Run pytest
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=self.capture_output,
            text=True
        )

        duration = time.time() - start_time

        # Parse output for test counts
        output = result.stdout + result.stderr
        test_count = self._parse_test_count(output)
        passed_count = test_count if result.returncode == 0 else 0
        failed_count = 0 if result.returncode == 0 else test_count

        return TestResult(
            category=category.value,
            passed=result.returncode == 0,
            duration_s=duration,
            test_count=test_count,
            passed_count=passed_count,
            failed_count=failed_count,
            output=output
        )

    def _parse_test_count(self, output: str) -> int:
        """Parse test count from pytest output"""
        # Look for "X passed" or "X failed" in output
        import re

        # Pattern: "5 passed in 0.23s"
        match = re.search(r'(\d+)\s+(?:passed|failed)', output)
        if match:
            return int(match.group(1))

        return 0

    def print_banner(self):
        """Print test runner banner"""
        print("=" * 70)
        print("🚀 NOCP Test Suite")
        print("=" * 70)
        print()

    def print_category_header(self, category: TestCategory, file_count: int):
        """Print category header"""
        print(f"\n{'='*20} {category.value.upper()} TESTS {'='*20}")
        print(f"Found {file_count} test file(s)")

    def print_result(self, result: TestResult):
        """Print test result with emoji"""
        emoji = "✅" if result.passed else "❌"
        status = "PASSED" if result.passed else "FAILED"

        print(f"\n{emoji} {result.category} tests {status}")
        print(f"   Tests: {result.test_count}")
        print(f"   Duration: {result.duration_s:.2f}s")

        if not result.passed and not self.verbose:
            print(f"\n{result.output}")

    def run_all_tests(
        self,
        category: Optional[TestCategory] = None
    ) -> TestSummary:
        """Run all discovered tests"""
        self.print_banner()

        discovered = self.discover_tests(category)

        if not discovered:
            print("❌ No tests found!")
            return TestSummary(
                total_tests=0,
                total_passed=0,
                total_failed=0,
                total_duration_s=0.0,
                success_rate=0.0,
                results=[]
            )

        categories_list = [cat.value for cat in discovered.keys()]
        print(f"📋 Test categories: {', '.join(categories_list)}\n")

        overall_success = True
        total_start = time.time()

        for cat_enum, test_files in discovered.items():
            self.print_category_header(cat_enum, len(test_files))

            result = self.run_pytest(test_files, cat_enum)
            self.results.append(result)

            self.print_result(result)

            if not result.passed:
                overall_success = False
                if self.fail_fast:
                    break

        total_duration = time.time() - total_start

        # Generate summary
        summary = self._generate_summary(total_duration)
        self._print_summary(summary)

        return summary

    def _generate_summary(self, total_duration: float) -> TestSummary:
        """Generate test summary"""
        total_tests = sum(r.test_count for r in self.results)
        total_passed = sum(r.passed_count for r in self.results)
        total_failed = sum(r.failed_count for r in self.results)

        success_rate = (
            (total_passed / total_tests * 100)
            if total_tests > 0
            else 0.0
        )

        return TestSummary(
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_duration_s=total_duration,
            success_rate=success_rate,
            results=self.results
        )

    def _print_summary(self, summary: TestSummary):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)

        print(f"\nTotal Tests:    {summary.total_tests}")
        print(f"Passed:         {summary.total_passed} ✅")
        print(f"Failed:         {summary.total_failed} ❌")
        print(f"Success Rate:   {summary.success_rate:.1f}%")
        print(f"Total Duration: {summary.total_duration_s:.2f}s")

        # Category breakdown
        print("\n📋 By Category:")
        for result in summary.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {result.category:12s} {status:8s} "
                  f"({result.test_count} tests, {result.duration_s:.2f}s)")

        print("\n" + "=" * 70)

    def save_json_report(self, output_path: Path, summary: TestSummary):
        """Save JSON test report"""
        report = asdict(summary)

        with output_path.open('w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 JSON report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="NOCP test runner with category organization"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["unit", "integration", "e2e", "performance"],
        help="Run tests from specific category only"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-x", "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Save JSON report to file"
    )

    args = parser.parse_args()

    # Convert category string to enum
    category = None
    if args.category:
        category = TestCategory(args.category)

    # Run tests
    runner = NOCPTestRunner(
        verbose=args.verbose,
        fail_fast=args.fail_fast
    )

    summary = runner.run_all_tests(category)

    # Save JSON report if requested
    if args.json:
        runner.save_json_report(args.json, summary)

    # Exit with appropriate code
    sys.exit(0 if summary.total_failed == 0 else 1)

if __name__ == "__main__":
    main()
```

**Benefits**:
- ✅ Organized test execution by category
- ✅ Beautiful progress output with emoji
- ✅ Detailed test summary
- ✅ JSON report generation
- ✅ Easy CI/CD integration

**Acceptance Criteria**:
- [ ] Test runner executable
- [ ] Categories: unit, integration, e2e, performance
- [ ] Progress tracking and summary
- [ ] JSON report generation
- [ ] Works in CI/CD

---

**D5.3.2: Organize Tests by Category** (3 hours)

Restructure tests directory with clear categories.

**New Structure**:

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── run_tests.py                   # Custom test runner
│
├── unit/                          # Fast, isolated tests (<100ms each)
│   ├── __init__.py
│   ├── test_config.py            # Configuration validation
│   ├── test_token_counter.py     # Token counting
│   ├── test_tool_executor.py     # Tool execution (mocked)
│   ├── test_context_manager.py   # Compression logic (mocked LLM)
│   ├── test_output_serializer.py # TOON encoding/decoding
│   ├── test_router.py            # Request routing
│   └── test_result.py            # Result pattern
│
├── integration/                   # Component integration (may call APIs)
│   ├── __init__.py
│   ├── test_agent_flow.py        # Act → Assess → Articulate flow
│   ├── test_compression_pipeline.py  # Full compression pipeline
│   ├── test_llm_integration.py   # LiteLLM integration
│   └── test_serialization_pipeline.py
│
├── e2e/                          # End-to-end workflows
│   ├── __init__.py
│   ├── test_complete_workflow.py # Full request → response
│   ├── test_gemini_integration.py # Real Gemini API calls
│   └── test_multi_turn.py        # Multi-turn conversations
│
└── performance/                  # Performance benchmarks
    ├── __init__.py
    ├── test_compression_speed.py
    ├── test_throughput.py
    └── test_memory_usage.py
```

**Migration Plan**:

1. Move existing `tests/core/test_act.py` → `tests/unit/test_tool_executor.py`
2. Create stub test files for each category
3. Add category markers in conftest.py
4. Update CI to run categories separately

**conftest.py enhancements**:

```python
import pytest

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers",
        "unit: Fast unit tests with no external dependencies"
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests that may call external APIs"
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end workflow tests"
    )
    config.addinivalue_line(
        "markers",
        "performance: Performance benchmarks"
    )
    config.addinivalue_line(
        "markers",
        "requires_api_key: Tests requiring GEMINI_API_KEY"
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests taking >1 second"
    )

# Run only fast tests by default
def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is specified"""
    if not config.getoption("--run-slow", default=False):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
```

**Benefits**:
- ✅ Clear test organization
- ✅ Faster test execution (run only unit tests)
- ✅ Better test maintenance
- ✅ Industry-standard structure

**Acceptance Criteria**:
- [ ] Tests reorganized into categories
- [ ] Each category has README explaining purpose
- [ ] Markers configured in conftest.py
- [ ] CI runs categories separately

---

**D5.3.3: Enhanced Test Fixtures** (2 hours)

Expand conftest.py with comprehensive fixtures.

```python
# tests/conftest.py additions

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import json

@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def temp_metrics_file(temp_dir):
    """Temporary metrics file"""
    metrics_file = temp_dir / "metrics.jsonl"
    yield metrics_file
    # Cleanup handled by temp_dir

@pytest.fixture
def sample_tool_output():
    """Sample tool output for testing"""
    return {
        "users": [
            {"id": f"user_{i}", "name": f"User {i}", "email": f"user{i}@example.com"}
            for i in range(10)
        ],
        "total": 10,
        "page": 1
    }

@pytest.fixture
def large_tool_output():
    """Large tool output for compression testing"""
    return {
        "records": [
            {
                "id": f"rec_{i}",
                "data": {
                    "field1": f"value_{i}",
                    "field2": i * 100,
                    "field3": ["item1", "item2", "item3"],
                },
                "metadata": {
                    "created": "2024-01-01T00:00:00Z",
                    "updated": "2024-01-01T00:00:00Z"
                }
            }
            for i in range(100)
        ]
    }

@pytest.fixture
def mock_gemini_response(mocker):
    """Mock Gemini API response"""
    mock_response = mocker.Mock()
    mock_response.text = "Mocked LLM response"
    mock_response.usage_metadata.prompt_token_count = 1000
    mock_response.usage_metadata.candidates_token_count = 50
    return mock_response

@pytest.fixture
def mock_litellm_response(mocker):
    """Mock LiteLLM API response"""
    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock()]
    mock_response.choices[0].message.content = "Mocked response"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 1000
    mock_response.usage.completion_tokens = 50
    return mock_response

@pytest.fixture
def config_minimal():
    """Minimal valid configuration"""
    from nocp.core.config import ProxyConfig
    return ProxyConfig(gemini_api_key="test-key")

@pytest.fixture
def config_all_features():
    """Configuration with all features enabled"""
    from nocp.core.config import ProxyConfig
    return ProxyConfig(
        gemini_api_key="test-key",
        enable_semantic_pruning=True,
        enable_knowledge_distillation=True,
        enable_history_compaction=True,
        enable_format_negotiation=True,
        default_compression_threshold=1000,
        target_compression_ratio=0.4,
    )

@pytest.fixture
def sample_context_data():
    """Sample ContextData for testing"""
    from nocp.models.schemas import ContextData, ToolResult

    return ContextData(
        tool_results=[
            ToolResult(
                tool_id="test_tool",
                success=True,
                data={"result": "test"},
                execution_time_ms=100
            )
        ],
        transient_context={"query": "test query"},
        max_tokens=10000
    )
```

**Benefits**:
- ✅ DRY test code
- ✅ Consistent test data
- ✅ Easier test writing
- ✅ Better test isolation

---

### Phase 5.4: CI/CD and Code Quality (Priority: HIGH)

**Effort**: 8-10 hours | **Impact**: High - Automated quality gates

#### Background

Rapydocs has comprehensive CI/CD:
- Tests on macOS ARM, Ubuntu, Windows
- Python 3.10, 3.11, 3.12 matrix
- Dependency caching
- Coverage reporting
- Pre-commit hooks

**NOCP Current State**: No GitHub Actions.

#### Deliverables

**D5.4.1: GitHub Actions CI/CD** (3 hours)

**File**: `.github/workflows/test.yml`

```yaml
name: CI Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  lint-and-type-check:
    name: Lint and Type Check
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run ruff (linting)
      run: |
        ruff check src/nocp tests/

    - name: Run black (formatting check)
      run: |
        black --check src/nocp tests/

    - name: Run mypy (type checking)
      run: |
        mypy src/nocp

  test:
    name: Test on ${{ matrix.os }} - Python ${{ matrix.python-version }}
    runs-on: ${{ matrix.os }}
    needs: lint-and-type-check

    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-${{ matrix.python-version }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-${{ matrix.python-version }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=nocp --cov-report=xml --cov-report=term

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v

    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-${{ matrix.os }}-py${{ matrix.python-version }}

  integration-test-with-api:
    name: Integration Tests (with API)
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run E2E tests with real API
      env:
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      run: |
        pytest tests/e2e/ -v -m "requires_api_key"

  build-check:
    name: Build Check
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - name: Install build tools
      run: |
        pip install build twine

    - name: Build package
      run: |
        python -m build

    - name: Check package
      run: |
        twine check dist/*
```

**Benefits**:
- ✅ Automated testing on every PR
- ✅ Cross-platform validation
- ✅ Early bug detection
- ✅ Code coverage tracking

**Acceptance Criteria**:
- [ ] CI runs on PR and push to main
- [ ] Tests on Ubuntu, macOS, Windows
- [ ] Python 3.10, 3.11, 3.12 tested
- [ ] Coverage uploaded to Codecov

---

**D5.4.2: Pre-commit Hooks** (1 hour)

**File**: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-json
      - id: check-toml
      - id: debug-statements
      - id: check-docstring-first

  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.15
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.0.0
          - types-pyyaml
        args: [--ignore-missing-imports]
```

**Setup**:

```bash
pip install pre-commit
pre-commit install
```

**Benefits**:
- ✅ Automatic code formatting before commit
- ✅ Catch issues early
- ✅ Consistent code style
- ✅ Less CI failures

**Acceptance Criteria**:
- [ ] Pre-commit config created
- [ ] Hooks installed in development setup
- [ ] Documentation updated with setup instructions

---

**D5.4.3: Enhanced pyproject.toml Tool Configuration** (1 hour)

```toml
# Enhanced tool configuration

[tool.ruff]
line-length = 100
target-version = "py311"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
    "SIM", # flake8-simplify
]
ignore = [
    "E501",  # line too long (handled by black)
]

[tool.ruff.per-file-ignores]
"tests/**/*.py" = ["S101"]  # Allow assert in tests

[tool.black]
line-length = 100
target-version = ["py311"]
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
check_untyped_defs = true
strict_equality = true
show_error_codes = true
show_column_numbers = true
show_error_context = true
pretty = true

[[tool.mypy.overrides]]
module = [
    "google.generativeai.*",
    "litellm.*",
    "structlog.*",
    "tenacity.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--strict-config",
    "--doctest-modules",
]
markers = [
    "unit: Fast unit tests with no external dependencies",
    "integration: Integration tests that may call external APIs",
    "e2e: End-to-end workflow tests",
    "performance: Performance benchmarks",
    "requires_api_key: Requires GEMINI_API_KEY environment variable",
    "slow: Tests taking >1 second (deselect with '-m \"not slow\"')",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]

[tool.coverage.run]
source = ["src/nocp"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__init__.py",
    "*/conftest.py",
]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if TYPE_CHECKING:",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == \"__main__\":",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
precision = 2
show_missing = true
skip_covered = false
skip_empty = true

[tool.coverage.html]
directory = "htmlcov"

[tool.coverage.xml]
output = "coverage.xml"
```

**Benefits**:
- ✅ Stricter linting rules
- ✅ Better type checking
- ✅ Organized test markers
- ✅ Comprehensive coverage config

---

### Phase 5.5: Documentation & Developer Experience (Priority: MEDIUM)

**Effort**: 8-10 hours | **Impact**: Medium-High - Better onboarding, maintenance

#### Deliverables

**D5.5.1: CLAUDE.md - AI Assistant Guide** (2 hours)

**File**: `CLAUDE.md`

```markdown
# Claude Code Developer Guide for NOCP

This file provides guidance to Claude Code and other AI assistants when working with the NOCP codebase.

## Quick Start

### Development Setup
```bash
# Clone and enter directory
git clone <repo> && cd nocp

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
│   ├── core/              # Core modules (agent, act, assess, articulate)
│   │   ├── agent.py       # Main orchestrator
│   │   ├── act.py         # Tool execution
│   │   ├── assess.py      # Context compression
│   │   ├── articulate.py  # Output serialization
│   │   └── config.py      # Configuration
│   ├── models/            # Pydantic schemas
│   │   ├── schemas.py     # Request/response models
│   │   ├── contracts.py   # Tool contracts
│   │   ├── context.py     # Context models
│   │   └── enums.py       # Configuration enums
│   ├── modules/           # Reusable modules
│   │   ├── context_manager.py
│   │   ├── output_serializer.py
│   │   ├── tool_executor.py
│   │   └── router.py
│   ├── llm/               # LLM integration
│   │   ├── client.py      # LiteLLM wrapper
│   │   └── router.py      # Model routing
│   ├── utils/             # Utilities
│   │   ├── logging.py     # Structured logging
│   │   ├── rich_logging.py # Rich console output
│   │   ├── token_counter.py
│   │   ├── error_handler.py
│   │   └── dependencies.py
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

## Architecture: Act-Assess-Articulate

NOCP implements a three-phase pipeline for efficient LLM interaction:

### 1. Act (Tool Execution)
- **File**: `src/nocp/core/act.py`
- **Purpose**: Execute tools with retry logic and timeout handling
- **Key Classes**: `ToolExecutor`, `ToolRequest`, `ToolResult`

### 2. Assess (Context Optimization)
- **File**: `src/nocp/core/assess.py`
- **Purpose**: Compress context to reduce token usage
- **Strategies**: Semantic Pruning, Knowledge Distillation, History Compaction
- **Key Classes**: `ContextManager`, `ContextData`, `OptimizedContext`

### 3. Articulate (Output Serialization)
- **File**: `src/nocp/core/articulate.py`
- **Purpose**: Serialize output efficiently (TOON or compact JSON)
- **Key Classes**: `OutputSerializer`, `SerializationRequest`

### Orchestrator
- **File**: `src/nocp/core/agent.py`
- **Purpose**: Coordinate all phases and handle errors
- **Key Class**: `HighEfficiencyProxyAgent`

## Development Guidelines

### Adding a New Tool

1. Define tool function with type hints
2. Register with ToolExecutor
3. Add tool schema for Gemini/LiteLLM
4. Write unit tests

```python
# In your module
from nocp.core.act import ToolExecutor

executor = ToolExecutor()

@executor.register_tool("my_tool")
def my_tool(param: str) -> dict:
    """Tool description"""
    return {"result": f"Processed {param}"}

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
# In src/nocp/core/assess.py
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
2. Environment variables (with `NOCP_` prefix)
3. `.env` file
4. `[tool.nocp]` in `pyproject.toml`
5. Hardcoded defaults

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
- [Component Specs](docs/04-COMPONENT-SPECS.md)
- [Testing Strategy](docs/05-TESTING-STRATEGY.md)
```

**Benefits**:
- ✅ Better AI assistant support
- ✅ Onboarding documentation
- ✅ Quick reference for developers

---

**D5.5.2: Component Documentation** (3 hours)

Create detailed documentation for compression strategies.

**File**: `docs/07-COMPRESSION-STRATEGIES.md`

```markdown
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
    {"id": "1", "name": "Alice", "email": "alice@ex.com", "created": "2024-01-01", "updated": "2024-01-01", ...},
    {"id": "2", "name": "Bob", "email": "bob@ex.com", "created": "2024-01-01", "updated": "2024-01-01", ...},
    // ... 100 more users
  ]
}
```

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

NOCP automatically calculates whether compression is worthwhile:

```python
compression_cost = (tokens_to_compress / 1_000_000) * student_model_cost_per_million
token_savings = original_tokens - compressed_tokens
main_model_savings = (token_savings / 1_000_000) * main_model_cost_per_million

if main_model_savings > compression_cost:
    # Compress
else:
    # Use original
```

## Combining Strategies

Strategies can be combined for maximum effect:

```python
config = ProxyConfig(
    enable_semantic_pruning=True,      # Remove redundancy
    enable_knowledge_distillation=True, # Summarize verbose outputs
    enable_history_compaction=True,     # Compress conversation history
)
```

**Order of operations**:
1. Semantic Pruning (structural optimization)
2. Knowledge Distillation (content summarization)
3. History Compaction (temporal optimization)

## Best Practices

### 1. Tool-Specific Thresholds

Set different thresholds for different tools:

```python
config = ProxyConfig()
config.set_tool_compression_threshold("database_query", 10_000)
config.set_tool_compression_threshold("web_search", 5_000)
```

### 2. Monitor Compression Ratios

Track compression effectiveness:

```python
metrics = agent.process_request(request)

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

Compare compressed vs uncompressed:

```python
# Process with compression
compressed_response = agent.process(request, enable_compression=True)

# Process without compression
uncompressed_response = agent.process(request, enable_compression=False)

# Compare quality and cost
print(f"Compressed quality: {evaluate_quality(compressed_response)}")
print(f"Cost savings: {compressed_response.cost_savings}")
```

## Troubleshooting

### Low Compression Ratios

**Problem**: Compression ratio >70% (less than 30% reduction)

**Solutions**:
- Lower compression threshold
- Enable additional strategies
- Check data for actual redundancy
- Increase target compression ratio

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
    raw_input_tokens: int              # Before compression
    compressed_input_tokens: int       # After compression
    input_compression_ratio: float     # Compressed / Raw

    compression_latency_ms: float      # Time spent compressing
    llm_inference_latency_ms: float    # Time spent in LLM

    compression_operations: List[str]  # Which strategies were used
    token_savings: int                 # Total tokens saved
```

Access via:

```python
response, metrics = agent.process_request(request)

print(f"Token savings: {metrics.raw_input_tokens - metrics.compressed_input_tokens:,}")
print(f"Compression ratio: {metrics.input_compression_ratio:.0%}")
print(f"Strategies used: {', '.join(metrics.compression_operations)}")
```
```

**Benefits**:
- ✅ Comprehensive strategy documentation
- ✅ Configuration examples
- ✅ Best practices
- ✅ Troubleshooting guide

---

### Phase 5.6: Advanced Error Handling (Priority: MEDIUM)

**Effort**: 6-8 hours | **Impact**: Medium - Better error handling, debugging

#### Deliverables

**D5.6.1: Result Pattern Implementation** (3 hours)

**File**: `src/nocp/models/result.py`

```python
"""Result type for explicit error handling (Rust-style)"""
from dataclasses import dataclass
from typing import Generic, TypeVar, Optional, List, Any, Callable

T = TypeVar('T')
E = TypeVar('E')

@dataclass
class Result(Generic[T]):
    """
    Result wrapper implementing "error as value" pattern.

    Inspired by Rust's Result<T, E> and functional programming.
    Eliminates exceptions for flow control.

    Examples:
        # Create successful result
        result = Result.ok({"user": "Alice"})

        # Create failed result
        result = Result.err("Database connection failed")

        # Check success
        if result.success:
            print(result.data)
        else:
            print(result.error)

        # Unwrap (raises if failed)
        data = result.unwrap()

        # Unwrap with default
        data = result.unwrap_or(default_value)

        # Chain operations
        result.map(process_data).map(transform_data)
    """

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

        # Validate invariants
        if self.success and self.data is None:
            raise ValueError("Successful result must have data")
        if not self.success and self.error is None:
            raise ValueError("Failed result must have error")

    def add_warning(self, warning: str) -> 'Result[T]':
        """Add a warning to the result"""
        if self.warnings is None:
            self.warnings = []
        self.warnings.append(warning)
        return self

    def unwrap(self) -> T:
        """
        Unwrap the result, raising exception if failed.

        Use only when you're certain the operation succeeded.
        Prefer unwrap_or() or explicit success checks.

        Raises:
            ValueError: If result is not successful
        """
        if not self.success:
            raise ValueError(f"Unwrap called on failed result: {self.error}")
        return self.data

    def unwrap_or(self, default: T) -> T:
        """Unwrap the result or return default value"""
        return self.data if self.success else default

    def unwrap_or_else(self, func: Callable[[], T]) -> T:
        """Unwrap or compute default via function"""
        return self.data if self.success else func()

    def map(self, func: Callable[[T], Any]) -> 'Result[Any]':
        """
        Apply function to data if successful.

        Returns new Result with transformed data or original error.
        """
        if self.success and self.data is not None:
            try:
                new_data = func(self.data)
                return Result(
                    success=True,
                    data=new_data,
                    warnings=self.warnings
                )
            except Exception as e:
                return Result(
                    success=False,
                    error=f"Map failed: {str(e)}",
                    warnings=self.warnings
                )

        return Result(
            success=False,
            error=self.error,
            warnings=self.warnings
        )

    def and_then(self, func: Callable[[T], 'Result[Any]']) -> 'Result[Any]':
        """
        Chain Result-returning functions (flatMap/bind).

        If successful, applies func to data and returns its Result.
        If failed, returns original error.
        """
        if self.success and self.data is not None:
            try:
                return func(self.data)
            except Exception as e:
                return Result.err(f"and_then failed: {str(e)}")

        return Result(success=False, error=self.error, warnings=self.warnings)

    def or_else(self, func: Callable[[str], 'Result[T]']) -> 'Result[T]':
        """
        Provide fallback if failed.

        If failed, calls func with error and returns its Result.
        If successful, returns original result.
        """
        if not self.success:
            try:
                return func(self.error)
            except Exception as e:
                return Result.err(f"or_else failed: {str(e)}")

        return self

    @classmethod
    def ok(cls, data: T) -> 'Result[T]':
        """Create successful result"""
        return cls(success=True, data=data)

    @classmethod
    def err(cls, error: str) -> 'Result[T]':
        """Create failed result"""
        return cls(success=False, error=error)

    def __repr__(self) -> str:
        if self.success:
            return f"Result.ok({self.data!r})"
        else:
            return f"Result.err({self.error!r})"
```

**Usage Examples**:

```python
# In tool_executor.py
def execute_tool(self, tool_name: str, **kwargs) -> Result[ToolResult]:
    """Execute tool with explicit error handling"""
    try:
        tool_func = self._tools.get(tool_name)
        if not tool_func:
            return Result.err(f"Tool '{tool_name}' not found")

        result = tool_func(**kwargs)
        return Result.ok(result)

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return Result.err(str(e))

# Usage
result = executor.execute_tool("search", query="test")

if result.success:
    tool_result = result.data
    process_tool_result(tool_result)
else:
    logger.error(f"Tool failed: {result.error}")
    # Handle error explicitly

# Or chain operations
executor.execute_tool("search", query="test") \
    .map(lambda r: r.data) \
    .map(process_data) \
    .unwrap_or(default_result)
```

**Benefits**:
- ✅ Explicit error handling (no silent failures)
- ✅ Chainable operations
- ✅ Better error propagation
- ✅ Functional programming style
- ✅ Type-safe error handling

**Acceptance Criteria**:
- [ ] Result class implemented with tests
- [ ] Key modules refactored to use Result
- [ ] Documentation with usage examples
- [ ] Migration guide for existing code

---

**D5.6.2: ErrorHandler Utilities** (2 hours)

**File**: `src/nocp/utils/error_handler.py`

```python
"""Centralized error handling utilities"""
from typing import Callable, TypeVar, Optional
from contextlib import contextmanager
import time
from functools import wraps
from ..utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class ErrorHandler:
    """Centralized error handling with logging and fallbacks"""

    @staticmethod
    def handle_with_fallback(
        operation: Callable[[], T],
        fallback: T,
        error_msg: str,
        log_level: str = "error"
    ) -> T:
        """
        Execute operation with fallback on error.

        Args:
            operation: Function to execute
            fallback: Value to return on error
            error_msg: Error message prefix
            log_level: Log level for errors

        Returns:
            Operation result or fallback value

        Example:
            result = ErrorHandler.handle_with_fallback(
                lambda: fetch_from_cache(),
                fallback=[],
                error_msg="Cache fetch failed"
            )
        """
        try:
            return operation()
        except Exception as e:
            getattr(logger, log_level)(f"{error_msg}: {e}")
            return fallback

    @staticmethod
    @contextmanager
    def log_duration(operation_name: str, log_level: str = "info"):
        """
        Context manager to log operation duration.

        Example:
            with ErrorHandler.log_duration("Database query"):
                result = db.query(...)
        """
        start = time.perf_counter()
        try:
            getattr(logger, log_level)(f"Starting {operation_name}")
            yield
        finally:
            duration = time.perf_counter() - start
            getattr(logger, log_level)(
                f"{operation_name} completed in {duration:.3f}s"
            )

    @staticmethod
    def retry_with_backoff(
        operation: Callable[[], T],
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        retryable_exceptions: tuple = (Exception,)
    ) -> T:
        """
        Retry operation with exponential backoff.

        Args:
            operation: Function to execute
            max_attempts: Maximum retry attempts
            backoff_factor: Multiplier for delay between retries
            initial_delay: Initial delay in seconds
            retryable_exceptions: Tuple of exceptions to retry on

        Returns:
            Operation result

        Raises:
            Last exception if all attempts fail

        Example:
            response = ErrorHandler.retry_with_backoff(
                lambda: requests.get(url),
                max_attempts=3,
                retryable_exceptions=(requests.Timeout, requests.ConnectionError)
            )
        """
        delay = initial_delay
        last_exception = None

        for attempt in range(max_attempts):
            try:
                return operation()
            except retryable_exceptions as e:
                last_exception = e

                if attempt == max_attempts - 1:
                    # Last attempt failed
                    raise

                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor

        # Should not reach here, but for type safety
        raise last_exception

    @staticmethod
    def ignore_errors(
        operation: Callable[[], T],
        error_msg: str = "Operation failed",
        log_level: str = "warning"
    ) -> Optional[T]:
        """
        Execute operation, ignoring all errors.

        Returns None on error. Use with caution!

        Example:
            # Non-critical cache write
            ErrorHandler.ignore_errors(
                lambda: cache.set(key, value),
                error_msg="Cache write failed (non-critical)"
            )
        """
        try:
            return operation()
        except Exception as e:
            getattr(logger, log_level)(f"{error_msg}: {e}")
            return None

def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0
):
    """
    Decorator for automatic retry with exponential backoff.

    Example:
        @with_retry(max_attempts=3)
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return ErrorHandler.retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                backoff_factor=backoff_factor,
                initial_delay=initial_delay
            )
        return wrapper
    return decorator
```

**Usage Examples**:

```python
# In agent.py
from ..utils.error_handler import ErrorHandler, with_retry

# Retry LLM calls
@with_retry(max_attempts=3, initial_delay=2.0)
def call_llm(self, prompt: str) -> LLMResponse:
    """Call LLM with automatic retries"""
    return self.llm_client.complete(prompt)

# Log duration
def process_request(self, request: AgentRequest) -> AgentResponse:
    with ErrorHandler.log_duration("Process request"):
        # Execute pipeline
        tool_results = self._execute_tools(request.tools)
        optimized = self._optimize_context(tool_results)
        response = self._call_llm(optimized)
        return response

# Fallback pattern
def get_cached_result(self, key: str) -> Optional[dict]:
    """Get from cache with fallback to None"""
    return ErrorHandler.handle_with_fallback(
        lambda: self.cache.get(key),
        fallback=None,
        error_msg="Cache lookup failed",
        log_level="warning"
    )
```

**Benefits**:
- ✅ Consistent error handling across codebase
- ✅ Automatic retries for transient failures
- ✅ Performance monitoring built-in
- ✅ Graceful degradation
- ✅ Centralized error logging

---

**D5.6.3: Enhanced Exceptions** (1 hour)

**File**: `src/nocp/exceptions.py` (enhancement)

```python
"""Enhanced exception classes with rich context"""
from typing import Any, Dict, Optional
from datetime import datetime

class ProxyAgentError(Exception):
    """Base exception with enhanced context and metadata"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None,
        recoverable: bool = False,
        user_message: Optional[str] = None
    ):
        """
        Initialize exception with context.

        Args:
            message: Technical error message for logs
            details: Additional context (dict for structured logging)
            retry_after: Seconds to wait before retrying (if applicable)
            recoverable: Whether error is recoverable with retry
            user_message: User-friendly error message
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.retry_after = retry_after
        self.recoverable = recoverable
        self.user_message = user_message or message
        self.timestamp = datetime.now()

    def __str__(self) -> str:
        parts = [self.message]

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"({details_str})")

        if self.retry_after:
            parts.append(f"[retry after {self.retry_after}s]")

        if self.recoverable:
            parts.append("[recoverable]")

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "retry_after": self.retry_after,
            "recoverable": self.recoverable,
            "user_message": self.user_message,
            "timestamp": self.timestamp.isoformat()
        }


class LLMError(ProxyAgentError):
    """LLM API errors with retry information"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None
    ):
        # Rate limits (429) are usually recoverable
        recoverable = status_code == 429

        # Generate user-friendly message
        if status_code == 429:
            user_message = "API rate limit exceeded. Please try again in a moment."
        elif status_code == 401:
            user_message = "API authentication failed. Please check your API key."
        elif status_code == 503:
            user_message = "Service temporarily unavailable. Please try again."
        else:
            user_message = "An error occurred while calling the LLM API."

        super().__init__(
            message=message,
            details=details or {},
            retry_after=retry_after,
            recoverable=recoverable,
            user_message=user_message
        )
        self.status_code = status_code


class ToolExecutionError(ProxyAgentError):
    """Tool execution errors"""

    def __init__(
        self,
        message: str,
        tool_id: str,
        details: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ):
        details = details or {}
        details.update({
            "tool_id": tool_id,
            "retry_count": retry_count
        })

        super().__init__(
            message=message,
            details=details,
            recoverable=retry_count < 3,  # Recoverable if retries remain
            user_message=f"Tool '{tool_id}' execution failed."
        )
        self.tool_id = tool_id
        self.retry_count = retry_count


class CompressionError(ProxyAgentError):
    """Context compression errors"""

    def __init__(
        self,
        message: str,
        strategy: str,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["strategy"] = strategy

        super().__init__(
            message=message,
            details=details,
            recoverable=True,  # Can fall back to uncompressed
            user_message="Context compression failed. Using uncompressed context."
        )
        self.strategy = strategy


class SerializationError(ProxyAgentError):
    """Output serialization errors"""

    def __init__(
        self,
        message: str,
        format: str,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["format"] = format

        super().__init__(
            message=message,
            details=details,
            recoverable=True,  # Can fall back to JSON
            user_message=f"Failed to serialize to {format}. Using fallback format."
        )
        self.format = format


class ConfigurationError(ProxyAgentError):
    """Configuration validation errors"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value

        super().__init__(
            message=message,
            details=details,
            recoverable=False,  # Config errors require fix
            user_message=f"Configuration error: {message}"
        )
        self.field = field
        self.value = value
```

**Usage**:

```python
# Raise with context
raise LLMError(
    "API request failed",
    status_code=429,
    retry_after=60.0,
    details={"model": "gemini-2.0-flash", "tokens": 10000}
)

# Catch and log
try:
    result = tool_executor.execute(tool)
except ToolExecutionError as e:
    logger.error(
        "tool_execution_failed",
        **e.to_dict()
    )

    # Show user-friendly message
    print(f"Error: {e.user_message}")

    # Check if recoverable
    if e.recoverable:
        # Retry
        pass
```

**Benefits**:
- ✅ Rich error context for debugging
- ✅ User-friendly error messages
- ✅ Retry guidance built-in
- ✅ Structured error logging
- ✅ Recoverable vs fatal distinction

---

## Implementation Summary

### Phase 5 Total Effort: 48-62 hours

**Priority Breakdown**:

- **HIGH Priority** (25-35 hours):
  - Configuration enhancements: 8-10 hours
  - Rich logging: 10-14 hours
  - Testing infrastructure: 12-15 hours
  - CI/CD: 8-10 hours

- **MEDIUM Priority** (15-20 hours):
  - Documentation: 8-10 hours
  - Error handling: 6-8 hours

### Key Benefits

1. **Developer Experience**: Rich console output, progress bars, organized tests
2. **Code Quality**: Automated linting, type checking, formatting
3. **Reliability**: Comprehensive testing, CI/CD, error handling
4. **Maintainability**: Better documentation, component loggers, configuration management
5. **Professional Appearance**: Makes NOCP feel production-ready

### Implementation Order

**Week 1**: HIGH priority items
1. Configuration enums and validators (Day 1-2)
2. PyProject.toml configuration (Day 2-3)
3. Rich logging integration (Day 3-5)

**Week 2**: HIGH priority items
1. Test runner and organization (Day 6-8)
2. CI/CD setup (Day 9-10)

**Week 3**: MEDIUM priority items
1. CLAUDE.md and documentation (Day 11-13)
2. Error handling improvements (Day 14-15)

### Success Metrics

- [ ] All configuration uses enums (type-safe)
- [ ] Rich console output for all user-facing operations
- [ ] >85% test coverage with organized test suite
- [ ] CI/CD runs on all PRs (Ubuntu, macOS, Windows)
- [ ] Pre-commit hooks prevent bad commits
- [ ] CLAUDE.md guides AI assistants effectively
- [ ] Result pattern used in critical paths
- [ ] Enhanced exceptions with user-friendly messages

---

## Phase 6: MDMAI-Inspired Enterprise Enhancements

**Status**: 🆕 NEW - Based on comprehensive analysis of the MDMAI (Multi-Domain Multi-Agent Intelligence) project
**Goal**: Transform NOCP into an enterprise-grade system with production patterns from MDMAI's sophisticated TTRPG assistant architecture.

**Context**: After analyzing the MDMAI project (MCP server with enterprise routing, hybrid search, defense-in-depth security), we identified critical enhancements that would elevate NOCP to production-grade quality with advanced reliability, performance, and security features.

**Total Estimated Effort**: 80-100 hours across 6 sub-phases

---

### Phase 6.1: Result Pattern & Advanced Error Handling (Priority: CRITICAL)

**Effort**: 12-15 hours | **Impact**: Critical - Type-safe error handling, better reliability

#### Background

MDMAI demonstrates superior error handling using the Result pattern (Returns library) instead of exceptions:
- Explicit error types in function signatures
- Composable error handling with `.bind()` and `.map()`
- No hidden control flow from exceptions
- Better testability without exception mocking

**NOCP Current State**: Traditional exception-based error handling throughout.

#### Deliverables

**D6.1.1: Result Pattern Implementation** (8 hours)

Install and configure the Returns library for type-safe error handling:

**Files**: `src/nocp/core/result.py`, `src/nocp/core/decorators.py`

Key components:
- `NOCPError` dataclass with ErrorKind enum
- `with_result` decorator for automatic exception wrapping
- Migration of core modules to Result pattern
- Backward compatibility during transition

**Benefits**:
- ✅ Explicit error handling in type signatures
- ✅ Composable operations with monadic patterns
- ✅ No hidden control flow
- ✅ Better testability
- ✅ Type-safe error propagation

**D6.1.2: Module Migration to Result Pattern** (4-7 hours)

Migrate critical modules to use Result pattern:
- Context Manager: Chain operations with Result monad
- Tool Executor: Return Result[ToolExecutionResult, NOCPError]
- Output Serializer: Explicit serialization failures
- Request Router: Type-safe routing errors

**Acceptance Criteria**:
- [ ] Returns library integrated
- [ ] Core modules use Result pattern
- [ ] Error types explicitly defined
- [ ] Tests updated for Result pattern
- [ ] Documentation includes Result pattern examples

---

### Phase 6.2: Enterprise Multi-Provider Orchestration (Priority: HIGH)

**Effort**: 15-18 hours | **Impact**: High - Resilience, cost optimization, intelligent routing

#### Background

MDMAI's enterprise router provides:
- 7 intelligent routing strategies (cost, speed, quality, reliability, load-balanced, adaptive, composite)
- Circuit breaker pattern for provider failures
- 4-tier fallback system (Primary → Secondary → Emergency → Local)
- Provider capability matching
- Real-time cost optimization

**NOCP Current State**: Basic LiteLLM support, no intelligent routing.

#### Deliverables

**D6.2.1: Provider Registry & Capabilities** (3 hours)

**File**: `src/nocp/providers/registry.py`

- Provider capability enum (function_calling, long_context, vision, etc.)
- ProviderConfig with costs, latency, reliability scores
- Registry for Gemini, OpenAI, Anthropic, local models

**D6.2.2: Intelligent Router** (6 hours)

**File**: `src/nocp/providers/intelligent_router.py`

- 7 routing strategies implementation
- Tier-based filtering (availability → capability → optimization)
- Composite scoring for balanced selection
- Routing history and state tracking
- Provider quality estimation

**D6.2.3: Circuit Breaker Implementation** (3 hours)

**File**: `src/nocp/providers/circuit_breaker.py`

- Circuit states: CLOSED, OPEN, HALF_OPEN
- Configurable failure/success thresholds
- Automatic recovery detection
- Per-provider circuit breakers

**D6.2.4: Fallback Manager** (3-6 hours)

**File**: `src/nocp/providers/fallback_manager.py`

- 4-tier fallback system
- Per-tier circuit breakers
- Automatic tier progression on failures
- Local model support as last resort

**Benefits**:
- ✅ 99.9%+ availability with fallback
- ✅ 30-50% cost reduction with intelligent routing
- ✅ Automatic failure recovery
- ✅ Provider-agnostic architecture
- ✅ Load balancing across providers

**Acceptance Criteria**:
- [ ] Provider registry with capability matching
- [ ] 7 routing strategies implemented
- [ ] Circuit breakers prevent cascade failures
- [ ] 4-tier fallback system operational
- [ ] Cost tracking and optimization metrics

---

### Phase 6.3: Advanced Three-Tier Caching (Priority: HIGH)

**Effort**: 12-15 hours | **Impact**: High - 40%+ performance improvement, cost reduction

#### Background

MDMAI's caching strategy:
- L1: In-memory cache (sub-1ms)
- L2: Redis distributed cache (1-5ms)
- L3: Persistent SQLite cache (10-50ms)
- Automatic tier promotion
- LRU eviction with category-based TTL

**NOCP Current State**: No caching implementation.

#### Deliverables

**D6.3.1: Cache Interface & Base** (2 hours)

**File**: `src/nocp/cache/base.py`

- CacheBackend abstract base class
- CacheEntry with metadata
- Async cache operations

**D6.3.2: L1 Memory Cache** (3 hours)

**File**: `src/nocp/cache/memory_cache.py`

- LRU eviction strategy
- Configurable size and TTL
- Hit/miss statistics
- Thread-safe operations

**D6.3.3: L2 Redis Cache** (3 hours)

**File**: `src/nocp/cache/redis_cache.py`

- Redis async client
- Distributed caching for multi-instance
- Configurable key prefixes
- Pickle serialization

**D6.3.4: L3 Persistent Cache** (2 hours)

**File**: `src/nocp/cache/persistent_cache.py`

- SQLite-based persistent storage
- Survives process restarts
- Index optimization
- Vacuum scheduling

**D6.3.5: Cache Manager** (2-5 hours)

**File**: `src/nocp/cache/cache_manager.py`

- Multi-tier coordination
- Automatic tier promotion
- Cache key generation
- Invalidation strategies

**Benefits**:
- ✅ 40%+ cache hit rate for repeated patterns
- ✅ Sub-millisecond response for cached items
- ✅ Reduced API costs
- ✅ Improved latency P50/P95
- ✅ Persistent cache survives restarts

**Acceptance Criteria**:
- [ ] Three-tier cache operational
- [ ] Cache hit rate metrics
- [ ] Automatic tier promotion
- [ ] Configuration for each tier
- [ ] Performance benchmarks show improvement

---

### Phase 6.4: Async & Concurrent Processing (Priority: MEDIUM)

**Effort**: 10-12 hours | **Impact**: Medium - 2-3x throughput improvement

#### Background

MDMAI's async architecture:
- Fully async pipeline
- Concurrent tool execution
- Thread pool for CPU-intensive operations
- Non-blocking I/O throughout

**NOCP Current State**: Synchronous processing.

#### Deliverables

**D6.4.1: Async Tool Executor** (4 hours)

**File**: `src/nocp/modules/async_tool_executor.py`

- Async tool registration and execution
- Concurrent batch execution
- Thread pool for sync tools
- Semaphore-based rate limiting

**D6.4.2: Async Context Manager** (3 hours)

**File**: `src/nocp/modules/async_context_manager.py`

- Async compression operations
- Concurrent output processing
- Cache-aware compression

**D6.4.3: Async Pipeline Orchestrator** (3-5 hours)

**File**: `src/nocp/core/async_agent.py`

- Full async request processing
- Parallel provider calls
- Non-blocking metrics logging
- Pipeline stage coordination

**Benefits**:
- ✅ 2-3x throughput improvement
- ✅ Better resource utilization
- ✅ Reduced latency for multi-tool requests
- ✅ Scalable to high concurrency

**Acceptance Criteria**:
- [ ] Async tool execution operational
- [ ] Concurrent compression working
- [ ] Full async pipeline tested
- [ ] Performance benchmarks show improvement

---

### Phase 6.5: Comprehensive Monitoring & Observability (Priority: MEDIUM)

**Effort**: 10-12 hours | **Impact**: Medium - Production visibility, debugging

#### Background

MDMAI's monitoring:
- Prometheus metrics
- OpenTelemetry tracing
- Health check endpoints
- Performance profiling
- Alert management

**NOCP Current State**: Basic JSONL metrics logging.

#### Deliverables

**D6.5.1: Metrics Collection** (4 hours)

**File**: `src/nocp/monitoring/metrics.py`

- Prometheus metrics (Counter, Histogram, Gauge, Summary)
- OpenTelemetry integration
- Request tracking
- Cost tracking

**D6.5.2: Health Monitoring** (2 hours)

**File**: `src/nocp/monitoring/health.py`

- Component health checks
- Overall system health
- Provider availability
- Cache backend status

**D6.5.3: Performance Profiler** (2 hours)

**File**: `src/nocp/monitoring/profiler.py`

- Code section profiling
- Bottleneck identification
- Memory profiling

**D6.5.4: Alert System** (2-4 hours)

**File**: `src/nocp/monitoring/alerting.py`

- Alert channels (Slack, email)
- Alert rules and thresholds
- Cooldown periods
- Alert history

**Benefits**:
- ✅ Real-time system visibility
- ✅ Performance bottleneck detection
- ✅ Proactive issue detection
- ✅ Production debugging capability

**Acceptance Criteria**:
- [ ] Prometheus metrics exposed
- [ ] Health endpoint returns status
- [ ] Alert rules configured
- [ ] Performance profiling operational

---

### Phase 6.6: Defense-in-Depth Security (Priority: HIGH)

**Effort**: 12-15 hours | **Impact**: High - Production security

#### Background

MDMAI's security:
- Multi-layer input validation
- Process sandboxing
- Resource limits
- Audit logging
- Path traversal prevention

**NOCP Current State**: Basic input validation.

#### Deliverables

**D6.6.1: Input Validation Layer** (4 hours)

**File**: `src/nocp/security/validation.py`

- SQL/command injection prevention
- Path traversal detection
- Size limits enforcement
- Parameter type validation

**D6.6.2: Sandboxing** (4 hours)

**File**: `src/nocp/security/sandbox.py`

- Resource limits (memory, CPU, file size)
- Process isolation
- Timeout enforcement
- Subprocess sandboxing

**D6.6.3: Audit Logging** (4-7 hours)

**File**: `src/nocp/security/audit.py`

- Cryptographically signed logs
- PII protection
- Integrity verification
- Structured audit events

**Benefits**:
- ✅ Protection against injection attacks
- ✅ Resource exhaustion prevention
- ✅ Audit trail for compliance
- ✅ Defense against malicious inputs

**Acceptance Criteria**:
- [ ] Input validation catches injections
- [ ] Resource limits enforced
- [ ] Audit logs cryptographically signed
- [ ] Security tests pass

---

### Phase 6.7: Advanced Testing Infrastructure (Priority: MEDIUM)

**Effort**: 8-10 hours | **Impact**: Medium - Quality assurance, reliability

#### Background

MDMAI's testing:
- Property-based testing with Hypothesis
- Load testing with Locust
- Benchmark suite
- Stateful testing

**NOCP Current State**: Basic unit tests.

#### Deliverables

**D6.7.1: Property-Based Testing** (3 hours)

**File**: `tests/property/`

- Hypothesis strategies
- Stateful testing for compression
- Invariant checking

**D6.7.2: Load Testing** (3 hours)

**File**: `tests/load/`

- Locust test scenarios
- Concurrent request testing
- Stress testing

**D6.7.3: Benchmark Suite** (2-4 hours)

**File**: `tests/benchmarks/`

- Compression benchmarks
- Serialization benchmarks
- Memory profiling

**Benefits**:
- ✅ Edge case discovery
- ✅ Performance regression detection
- ✅ Scalability validation
- ✅ Memory leak detection

**Acceptance Criteria**:
- [ ] Property tests find no violations
- [ ] Load tests meet performance targets
- [ ] Benchmarks establish baselines
- [ ] No performance regressions

---

## Phase 6 Implementation Summary

### Total Effort: 80-100 hours

**Priority Breakdown**:

- **CRITICAL** (12-15 hours):
  - Result pattern implementation

- **HIGH** (39-48 hours):
  - Multi-provider orchestration: 15-18 hours
  - Three-tier caching: 12-15 hours
  - Security hardening: 12-15 hours

- **MEDIUM** (28-36 hours):
  - Async processing: 10-12 hours
  - Monitoring & observability: 10-12 hours
  - Advanced testing: 8-10 hours

### Key Benefits

1. **Reliability**: Result pattern, circuit breakers, fallback system
2. **Performance**: 3-tier caching, async processing, 2-3x throughput
3. **Cost Optimization**: Intelligent routing, 30-50% cost reduction
4. **Security**: Defense-in-depth, sandboxing, audit logging
5. **Observability**: Metrics, tracing, profiling, alerting
6. **Quality**: Property-based testing, load testing, benchmarks

### Implementation Timeline

**Month 1**: Foundation (Critical + High Priority)
- Week 1-2: Result pattern migration
- Week 3-4: Multi-provider orchestration

**Month 2**: Performance & Security
- Week 5-6: Three-tier caching
- Week 7-8: Security hardening

**Month 3**: Polish & Testing
- Week 9-10: Async processing
- Week 11: Monitoring & observability
- Week 12: Advanced testing

### Success Metrics

- [ ] 99.9% availability with fallback
- [ ] 40%+ cache hit rate
- [ ] 30-50% cost reduction
- [ ] P95 latency < 3 seconds
- [ ] Zero security vulnerabilities
- [ ] 90%+ test coverage
- [ ] All critical paths use Result pattern

### Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Result pattern migration complexity | Medium | High | Gradual migration, backward compatibility |
| Provider API changes | Medium | Medium | Abstract provider interface, version pinning |
| Cache coherency issues | Low | Medium | TTL-based invalidation, versioned keys |
| Async migration bugs | Medium | High | Comprehensive testing, gradual rollout |
| Security false positives | Low | Low | Configurable validation rules |

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
- [ ] **Phase 5.1-5.3 complete** (Configuration, Logging, Testing)

### Production Release (v1.0.0)
- [ ] All Phase 4 milestones complete
- [ ] Async support fully tested
- [ ] Caching layer operational
- [ ] Security audit complete
- [ ] Performance meets all KPIs
- [ ] **Phase 5.4-5.6 complete** (CI/CD, Documentation, Error Handling)

### Enterprise Release (v2.0.0)
- [ ] **Phase 6.1 complete** (Result pattern implementation)
- [ ] **Phase 6.2 complete** (Multi-provider orchestration with circuit breakers)
- [ ] **Phase 6.3 complete** (Three-tier caching operational)
- [ ] **Phase 6.4 complete** (Async processing pipeline)
- [ ] **Phase 6.5 complete** (Monitoring & observability)
- [ ] **Phase 6.6 complete** (Defense-in-depth security)
- [ ] **Phase 6.7 complete** (Advanced testing infrastructure)
- [ ] 99.9% availability demonstrated
- [ ] 40%+ cache hit rate achieved
- [ ] 30-50% cost reduction verified
- [ ] P95 latency < 3 seconds
- [ ] Security audit passed
- [ ] 90%+ test coverage

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TOON library unavailable | Medium | High | Implement TOON encoder from spec |
| LiteLLM API changes | Low | Medium | Pin versions, monitor changelog |
| Compression overhead > savings | Medium | High | Cost-benefit checks, fallback to raw |
| uv installation fails | Low | High | Fallback to pip, clear error messages |
| Token counting inaccurate | Medium | Medium | Validate against actual usage, adjust |
| **Phase 5 scope creep** | **Medium** | **Medium** | **Time-box each deliverable, prioritize HIGH items** |
| **Rich library compatibility** | **Low** | **Low** | **Test on Windows, fallback to basic output** |
| **Test migration effort** | **Medium** | **Medium** | **Migrate incrementally, keep existing tests working** |
| **Phase 6: Result pattern migration** | **Medium** | **High** | **Gradual migration, maintain backward compatibility** |
| **Phase 6: Provider API changes** | **Medium** | **Medium** | **Abstract provider interface, version pinning** |
| **Phase 6: Cache coherency** | **Low** | **Medium** | **TTL-based invalidation, versioned cache keys** |
| **Phase 6: Async migration bugs** | **Medium** | **High** | **Comprehensive testing, gradual rollout** |
| **Phase 6: Security false positives** | **Low** | **Low** | **Configurable validation rules** |
| **Phase 6: Monitoring overhead** | **Low** | **Low** | **Configurable metrics, sampling rates** |

---

**Next**: See `04-COMPONENT-SPECS.md` for detailed implementation specifications.

**Phase 5 Reference**: See this document (03-DEVELOPMENT-ROADMAP.md) for comprehensive Phase 5 details.
