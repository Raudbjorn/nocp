# nocp - High-Efficiency LLM Proxy Agent

A Python-based middleware orchestration layer designed to optimize token costs and performance when integrating large-context LLM models (e.g., Gemini 2.5 Flash).

## 🎯 Overview

The **nocp** (No-Cost Proxy) agent addresses the "Paradox of Abundance" where vast context windows create economic penalties for inefficient data loading. It implements three core optimization components:

1. **Act** (Tool Executor) - Execute external tools and functions
2. **Assess** (Context Manager) - Compress input context (50-70% reduction target)
3. **Articulate** (Output Serializer) - Optimize output with TOON format (30-60% reduction target)

## 🚀 Quick Start

### Installation

The project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. The setup is completely automated:

```bash
# Clone the repository
git clone https://github.com/Raudbjorn/nocp
cd nocp

# Setup (automatically installs uv if needed)
./nocp setup

# Verify installation
./nocp --version
```

### Run the Demo

```bash
./nocp run examples/basic_usage.py
```

## 📊 Key Features

### Input Optimization (Assess Module)
- **Semantic Pruning**: Extract top-k relevant chunks from large datasets
- **Knowledge Distillation**: Summarize verbose outputs using lightweight LLMs
- **History Compaction**: Roll-up summarization of conversation history
- **Target**: 50-70% token reduction

### Output Optimization (Articulate Module)
- **TOON Serialization**: Token-optimized format for tabular data
- **Format Negotiation**: Automatic selection of optimal format
- **Compact JSON**: Minimal whitespace for nested structures
- **Target**: 30-60% token reduction

### Tool Execution (Act Module)
- **Flexible Registration**: Decorator-based tool registration
- **Retry Logic**: Exponential backoff for transient failures
- **Timeout Handling**: Prevent long-running operations
- **Async Support**: Concurrent tool execution

## 📚 Documentation

Comprehensive specification documentation is available in the `docs/` directory:

- [00-PROJECT-OVERVIEW.md](docs/00-PROJECT-OVERVIEW.md) - Project goals and architecture
- [01-ARCHITECTURE.md](docs/01-ARCHITECTURE.md) - Detailed system architecture
- [02-API-CONTRACTS.md](docs/02-API-CONTRACTS.md) - API contracts and data schemas
- [03-DEVELOPMENT-ROADMAP.md](docs/03-DEVELOPMENT-ROADMAP.md) - Phased development plan
- [04-COMPONENT-SPECS.md](docs/04-COMPONENT-SPECS.md) - Implementation specifications
- [05-TESTING-STRATEGY.md](docs/05-TESTING-STRATEGY.md) - Testing approach
- [06-DEPLOYMENT.md](docs/06-DEPLOYMENT.md) - Deployment and operations guide

## 🛠️ Development

### Commands

```bash
./nocp --help              # Show all commands
./nocp setup              # Install dependencies
./nocp run <script>       # Run Python script
./nocp test               # Run tests
./nocp shell              # Interactive Python shell
./nocp health             # System health check
./nocp info               # Project information
```

## 📈 Performance Targets

| Metric | Target |
|--------|--------|
| Input Token Reduction | 50-70% |
| Output Token Reduction | 30-60% |
| Compression Overhead | <200ms |
| Cost Reduction | 40-60% |

## 🗺️ Roadmap

### Phase 1: MVP (Current) ✅
- ✅ Bootstrap infrastructure with uv
- ✅ Act module (tool execution)
- ✅ Basic Assess module (semantic pruning)
- ✅ Articulate module with TOON support
- ✅ End-to-end proof of concept

---

**Version**: 0.1.0
**Status**: MVP Development
**Last Updated**: 2025-11-18