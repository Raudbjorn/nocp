# LLM Proxy Agent - Project Overview

## Executive Summary

The LLM Proxy Agent is a Python-based middleware orchestration layer designed to optimize token costs and performance when integrating large-context LLM models (e.g., Gemini 2.5 Flash). The system addresses the "Paradox of Abundance" where vast context windows create economic penalties for inefficient data loading.

## Core Value Proposition

- **Cost Optimization**: 50-70% reduction in input tokens, 30-60% reduction in output tokens
- **Performance**: Intelligent routing and compression without sacrificing quality
- **Flexibility**: Multi-cloud LLM provider support via LiteLLM integration
- **Reliability**: Pydantic-based data contracts and structured validation

## System Architecture (High-Level)

The system implements the **Orchestrator-Worker Pattern** with three core components:

```
┌─────────────────────────────────────────────────────┐
│                    Client Request                    │
└──────────────────────┬──────────────────────────────┘
                       ▼
            ┌──────────────────────┐
            │  Proxy Agent Entry   │
            │   (Orchestrator)     │
            └──────────┬───────────┘
                       ▼
         ┌─────────────┴─────────────┐
         ▼                           ▼
  ┌─────────────┐            ┌─────────────┐
  │    ACT      │            │   ASSESS    │
  │   (Tool     │───────────▶│  (Context   │
  │  Executor)  │            │  Manager)   │
  └─────────────┘            └──────┬──────┘
                                    ▼
                            ┌───────────────┐
                            │  Main LLM     │
                            │  (Reasoning)  │
                            └───────┬───────┘
                                    ▼
                            ┌───────────────┐
                            │  ARTICULATE   │
                            │   (Output     │
                            │ Serializer)   │
                            └───────┬───────┘
                                    ▼
                            ┌───────────────┐
                            │    Response   │
                            └───────────────┘
```

## Technology Stack

### Core Dependencies
- **Python 3.11+**: Base runtime
- **uv**: Astral's fast Python package manager (hidden from user)
- **Pydantic v2**: Data validation and schema generation
- **LiteLLM**: Multi-provider LLM gateway
- **python-toon**: Token-optimized serialization format

### Development Tools
- pytest: Testing framework
- ruff: Fast Python linter/formatter
- mypy: Static type checking

## Development Philosophy

### 1. Spec-Driven Development
All components are specified before implementation with clear:
- Input/Output contracts (Pydantic models)
- Success criteria and KPIs
- Integration points
- Test scenarios

### 2. Seamless Tooling (uv Integration)
- Single executable entry point (`nocp`)
- Automatic uv installation if missing
- Zero user exposure to underlying package management
- Cross-platform compatibility

### 3. Incremental Validation
Each component includes:
- Unit tests
- Integration tests
- Benchmarking harness
- Token usage validation

## Project Goals

### Phase 1: MVP (Weeks 1-3)
- ✅ Bootstrap infrastructure with uv
- ✅ Act module (tool execution)
- ✅ Basic Assess module (semantic pruning)
- ✅ Articulate module with TOON support
- ✅ End-to-end proof of concept

### Phase 2: Optimization (Weeks 4-6)
- Knowledge distillation via Student Summarizer
- Format negotiation layer
- Conversation history compaction
- Performance benchmarking suite

### Phase 3: Production Readiness (Weeks 7-9)
- Structured logging and observability
- Multi-cloud routing strategies
- Contextual drift detection
- Comprehensive documentation

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Input Token Reduction | 50-70% | Raw vs Compressed Context |
| Output Token Reduction | 30-60% | JSON vs TOON Serialization |
| Compression Overhead | <200ms | p95 latency for compression |
| End-to-End Latency | <2x baseline | With all optimizations enabled |
| Cost Reduction | 40-60% | Total token cost per transaction |

## Repository Structure

```
nocp/
├── docs/                    # Specification documentation
├── src/
│   └── nocp/
│       ├── __init__.py
│       ├── __main__.py      # Entry point (proxies uv)
│       ├── bootstrap.py     # uv auto-installer
│       ├── core/
│       │   ├── act.py       # Tool Executor
│       │   ├── assess.py    # Context Manager
│       │   └── articulate.py # Output Serializer
│       ├── models/          # Pydantic schemas
│       ├── serializers/     # TOON implementation
│       └── utils/           # Shared utilities
├── tests/                   # Test suite
├── benchmarks/              # Performance benchmarks
├── pyproject.toml          # Project metadata (uv-compatible)
└── README.md               # User-facing documentation
```

## Getting Started (User Perspective)

```bash
# Single command bootstrap
./nocp setup

# Runs tool with automatic dependency resolution
./nocp run my_task.py

# Benchmarking
./nocp benchmark --baseline
```

The `nocp` executable handles all uv operations transparently.

## Next Steps

1. Review complete specification documentation (docs/01-06)
2. Execute Phase 1 development roadmap
3. Validate MVP with benchmark suite
4. Iterate based on token efficiency metrics

---

**Document Version**: 1.0
**Last Updated**: 2025-11-18
**Status**: ✅ APPROVED FOR DEVELOPMENT
