#!/bin/bash

# NOCP - High-Efficiency LLM Proxy Agent Build Script
# Unified build system with automatic package manager detection
# Comprehensive git/GitHub status monitoring and CI/CD integration

# Check bash version (requires bash 4.3+ for namerefs)
if [ "${BASH_VERSINFO[0]}" -lt 4 ] || ([ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 3 ]); then
    echo "Error: This script requires bash 4.3 or later (found ${BASH_VERSION})" >&2
    echo "On macOS, install with: brew install bash" >&2
    exit 1
fi

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis for better UX
ROCKET="🚀"
CHECK="✅"
CROSS="❌"
WARNING="⚠️"
GEAR="⚙️"
PACKAGE="📦"
TEST="🧪"
CLEAN="🧹"
SEARCH="🔍"
DOCS="📚"

# Print functions
print_header() {
    echo -e "\n${PURPLE}╔═══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                     NOCP Build System - Token Optimization                    ║${NC}"
    echo -e "${PURPLE}╚═══════════════════════════════════════════════════════════════════════════════╝${NC}"

    # Show git/GitHub status warnings
    check_git_status
}

print_section() {
    echo -e "\n${CYAN}${GEAR} $1${NC}"
    echo -e "${CYAN}$(printf '%.0s─' {1..80})${NC}"
}

print_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

print_error() {
    echo -e "${RED}${CROSS} $1${NC}" >&2
}

print_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

print_info() {
    echo -e "${BLUE}${GEAR} $1${NC}"
}

# Command existence check
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Directory detection
get_script_dir() {
    cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd
}

# Git repository status check
check_git_status() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        return 0  # Not a git repo, skip checks
    fi

    local warnings=()

    # Check for uncommitted changes
    local uncommitted=$(git status --porcelain 2>/dev/null | wc -l)
    if [ "$uncommitted" -gt 20 ]; then
        warnings+=("🔄 You have $uncommitted uncommitted changes - consider committing or stashing")
    elif [ "$uncommitted" -gt 5 ]; then
        warnings+=("📝 You have $uncommitted uncommitted changes")
    fi

    # Check for unpushed commits and branch divergence
    local current_branch=$(git branch --show-current 2>/dev/null || echo "")
    if [ -n "$current_branch" ]; then
        local unpushed=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "0")
        if [ "$unpushed" -gt 0 ]; then
            warnings+=("📤 You have $unpushed unpushed commits on branch '$current_branch'")
        fi

        # Check if branch can be merged with main/master (quick check)
        check_merge_status warnings "$current_branch"

        # Check if branch is significantly behind main/master
        check_branch_divergence warnings "$current_branch"
    fi

    # Check for unmerged pull requests (if gh CLI is available)
    if command_exists gh; then
        check_github_status warnings
    fi

    # Display warnings if any
    if [ ${#warnings[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}${WARNING} Git Status Notifications:${NC}"
        for warning in "${warnings[@]}"; do
            echo -e "  ${YELLOW}$warning${NC}"
        done
        echo ""
    fi
}

# Check if current branch can merge cleanly with main/master
check_merge_status() {
    local -n warnings_ref=$1
    local current_branch=$2

    # Skip if we're on main/master
    if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
        return 0
    fi

    # Find the default branch (main or master)
    local default_branch=""
    if git show-ref --verify --quiet refs/heads/main; then
        default_branch="main"
    elif git show-ref --verify --quiet refs/heads/master; then
        default_branch="master"
    else
        return 0  # No default branch found
    fi

    # Quick merge conflict check (this is fast)
    local merge_base=$(git merge-base "$current_branch" "$default_branch" 2>/dev/null || echo "")
    if [ -n "$merge_base" ]; then
        # Check if there are conflicting files (this is the expensive part, so we limit it)
        local conflicts=$(git merge-tree "$merge_base" "$current_branch" "$default_branch" 2>/dev/null | grep -c "<<<<<<< " 2>/dev/null || echo "0")
        if [[ -n "$conflicts" && "$conflicts" -gt 0 ]]; then
            warnings_ref+=("⚠️ Branch '$current_branch' may have merge conflicts with '$default_branch' ($conflicts potential conflicts)")
        fi
    fi
}

# Check branch divergence from main/master
check_branch_divergence() {
    local -n warnings_ref=$1
    local current_branch=$2

    # Skip if we're on main/master
    if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
        return 0
    fi

    # Find the default branch
    local default_branch=""
    if git show-ref --verify --quiet refs/heads/main; then
        default_branch="main"
    elif git show-ref --verify --quiet refs/heads/master; then
        default_branch="master"
    else
        return 0
    fi

    # Check how far behind we are (this is very fast)
    local behind=$(git rev-list --count HEAD.."$default_branch" 2>/dev/null || echo "0")
    local ahead=$(git rev-list --count "$default_branch"..HEAD 2>/dev/null || echo "0")

    if [ "$behind" -gt 20 ]; then
        warnings_ref+=("📉 Branch '$current_branch' is $behind commits behind '$default_branch' - consider rebasing")
    elif [ "$behind" -gt 5 ]; then
        warnings_ref+=("📋 Branch '$current_branch' is $behind commits behind '$default_branch'")
    fi

    # Check for very long-running branches
    local days_old=$(git log --format="%ct" -1 "$default_branch" 2>/dev/null)
    local branch_base=$(git merge-base "$current_branch" "$default_branch" 2>/dev/null)
    if [ -n "$days_old" ] && [ -n "$branch_base" ]; then
        local base_time=$(git log --format="%ct" -1 "$branch_base" 2>/dev/null || echo "$days_old")
        local days_since=$(( ($(date +%s) - base_time) / 86400 ))
        if [ "$days_since" -gt 30 ]; then
            warnings_ref+=("📅 Branch '$current_branch' diverged $days_since days ago - consider updating or merging")
        fi
    fi
}

# GitHub CLI integration for PR checks
check_github_status() {
    local -n warnings_ref=$1

    # Check if we're in a GitHub repo
    local github_repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || echo "")
    if [ -z "$github_repo" ]; then
        return 0  # Not a GitHub repo or not authenticated
    fi

    # Check for open pull requests with detailed status
    local open_prs=$(gh pr list --state open --json number,title,author,isDraft,mergeable,reviewDecision 2>/dev/null)
    local pr_count=$(echo "$open_prs" | jq length 2>/dev/null || echo "0")

    if [ "$pr_count" -gt 0 ]; then
        # Count mergeable vs problematic PRs
        local mergeable_count=$(echo "$open_prs" | jq '[.[] | select(.mergeable == "MERGEABLE")] | length' 2>/dev/null || echo "0")
        local conflicted_count=$(echo "$open_prs" | jq '[.[] | select(.mergeable == "CONFLICTING")] | length' 2>/dev/null || echo "0")
        local draft_count=$(echo "$open_prs" | jq '[.[] | select(.isDraft == true)] | length' 2>/dev/null || echo "0")

        warnings_ref+=("🔀 There are $pr_count open pull request(s) in $github_repo")

        if [ "$conflicted_count" -gt 0 ]; then
            warnings_ref+=("⚠️ $conflicted_count PR(s) have merge conflicts")
        fi

        if [ "$draft_count" -gt 0 ]; then
            warnings_ref+=("📝 $draft_count draft PR(s) not ready for review")
        fi

        # Show specific PRs if not too many
        if [ "$pr_count" -le 5 ]; then
            local pr_info=$(echo "$open_prs" | jq -r '.[] | "  • #\(.number): \(.title) (@\(.author.login))" + (if .isDraft then " [DRAFT]" else "" end) + (if .mergeable == "CONFLICTING" then " [CONFLICTS]" else "" end)' 2>/dev/null | head -3)
            if [ -n "$pr_info" ]; then
                warnings_ref+=("$pr_info")
            fi
        fi
    fi

    # Check for PRs that need review (assigned to you)
    local review_prs=$(gh pr list --state open --review-requested @me --json number 2>/dev/null | jq length 2>/dev/null || echo "0")
    if [[ -n "$review_prs" && "$review_prs" -gt 0 ]]; then
        warnings_ref+=("👀 You have $review_prs pull request(s) awaiting your review")
    fi

    # Check for failed CI/CD runs on current branch
    local current_branch=$(git branch --show-current 2>/dev/null || echo "")
    if [ -n "$current_branch" ]; then
        local failed_runs=$(gh run list --branch "$current_branch" --status failure --limit 5 --json conclusion 2>/dev/null | jq length 2>/dev/null || echo "0")
        if [ "$failed_runs" -gt 0 ]; then
            warnings_ref+=("❌ Recent CI/CD failures on branch '$current_branch' - check 'gh run list'")
        fi
    fi
}

# Quick status command
show_git_status() {
    print_section "Repository Status"

    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_warning "Not in a git repository"
        return 0
    fi

    # Basic git status
    echo -e "${BLUE}Git Status:${NC}"
    local current_branch=$(git branch --show-current 2>/dev/null || echo "detached")
    local uncommitted=$(git status --porcelain 2>/dev/null | wc -l)
    local unpushed=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "unknown")

    echo -e "  Branch: ${CYAN}$current_branch${NC}"
    echo -e "  Uncommitted changes: ${CYAN}$uncommitted${NC}"
    echo -e "  Unpushed commits: ${CYAN}$unpushed${NC}"

    # GitHub status if available
    if command_exists gh; then
        local github_repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || echo "")
        if [ -n "$github_repo" ]; then
            echo -e "\n${BLUE}GitHub Status (${CYAN}$github_repo${BLUE}):${NC}"

            # Pull requests
            local open_prs=$(gh pr list --state open --json number,title,author 2>/dev/null)
            local pr_count=$(echo "$open_prs" | jq length 2>/dev/null || echo "0")
            echo -e "  Open pull requests: ${CYAN}$pr_count${NC}"

            if [ "$pr_count" -gt 0 ] && [ "$pr_count" -le 5 ]; then
                echo "$open_prs" | jq -r '.[] | "    • #\(.number): \(.title) (@\(.author.login))"' 2>/dev/null | head -3
            fi

            # Issues
            local open_issues=$(gh issue list --state open --json number 2>/dev/null | jq length 2>/dev/null || echo "0")
            echo -e "  Open issues: ${CYAN}$open_issues${NC}"

            # Recent workflow runs
            local recent_runs=$(gh run list --limit 3 --json status,conclusion,workflowName 2>/dev/null)
            if [ -n "$recent_runs" ] && [ "$recent_runs" != "null" ]; then
                echo -e "  Recent CI/CD runs:"
                echo "$recent_runs" | jq -r '.[] | "    • \(.workflowName): \(.conclusion // .status)"' 2>/dev/null
            fi
        else
            echo -e "\n${YELLOW}  Not authenticated with GitHub CLI or not a GitHub repo${NC}"
        fi
    else
        echo -e "\n${YELLOW}  GitHub CLI (gh) not available for enhanced status${NC}"
    fi
}

# Get project root
PROJECT_ROOT=$(get_script_dir)
cd "$PROJECT_ROOT"

# Package manager detection
detect_python_manager() {
    if command_exists uv; then
        echo "uv"
    elif command_exists poetry; then
        echo "poetry"
    else
        echo "pip"
    fi
}

# Setup Python virtual environment
setup_python_env() {
    local manager=$1
    print_info "Setting up Python environment with $manager"

    case $manager in
        uv)
            if [ ! -d ".venv" ]; then
                print_info "Creating virtual environment with uv..."
                uv venv
            fi
            print_info "Installing Python dependencies..."
            if ! uv pip install -e ".[dev]" --quiet 2>&1; then
                print_warning "Failed to install with dev extras, trying base installation..."
                if ! uv pip install -e . --quiet 2>&1; then
                    print_error "Failed to install Python dependencies with uv"
                    exit 1
                fi
            fi
            ;;
        poetry)
            print_info "Installing Python dependencies with Poetry..."
            poetry install --with dev --quiet
            ;;
        pip)
            if [ ! -d ".venv" ]; then
                print_info "Creating virtual environment..."
                python3 -m venv .venv
            fi
            source .venv/bin/activate
            print_info "Installing Python dependencies with pip..."
            pip install --upgrade pip --quiet
            if ! pip install -e ".[dev]" --quiet 2>&1; then
                print_warning "Failed to install with dev extras, trying base installation..."
                if ! pip install -e . --quiet 2>&1; then
                    print_error "Failed to install Python dependencies with pip"
                    exit 1
                fi
            fi
            ;;
    esac
}

# Run Python command with detected manager
run_python_cmd() {
    local cmd="$1"
    local manager=$(detect_python_manager)

    case $manager in
        uv)
            if [ -f ".venv/bin/activate" ]; then
                source .venv/bin/activate && eval "$cmd"
            else
                uv run $cmd
            fi
            ;;
        poetry)
            poetry run $cmd
            ;;
        pip)
            if [ -f ".venv/bin/activate" ]; then
                source .venv/bin/activate && eval "$cmd"
            else
                eval "$cmd"
            fi
            ;;
    esac
}

# Install dependencies
install_dependencies() {
    print_section "Installing Dependencies"

    local python_manager=$(detect_python_manager)

    print_info "Detected Python manager: $python_manager"

    # Python dependencies
    setup_python_env "$python_manager"
    print_success "Python dependencies installed"

    # Optional: Install litellm if requested
    if [ "${INSTALL_LITELLM:-false}" = "true" ]; then
        print_info "Installing optional litellm dependencies..."
        case $python_manager in
            uv)
                uv pip install litellm --quiet
                ;;
            poetry)
                poetry add litellm
                ;;
            pip)
                source .venv/bin/activate
                pip install litellm --quiet
                ;;
        esac
        print_success "LiteLLM installed"
    fi
}

# Build and validate
build_all() {
    print_section "Building NOCP"

    print_info "Running Python type checks..."
    run_python_cmd "mypy src/nocp --ignore-missing-imports" && \
        print_success "Type checks passed" || \
        print_warning "Type checks completed with warnings"

    print_info "Running Python linting..."
    run_python_cmd "ruff check src/nocp tests" && \
        print_success "Linting passed" || \
        print_warning "Linting completed with warnings"

    print_success "NOCP build complete"
}

# Testing
run_tests() {
    print_section "Running Tests"

    print_info "Running Python tests with coverage..."
    run_python_cmd "pytest tests/ -v --cov=src/nocp --cov-report=term-missing"
    print_success "Tests completed"
}

run_unit_tests() {
    print_section "Running Unit Tests"

    print_info "Running unit tests..."
    run_python_cmd "pytest tests/core/ tests/models/ -v"
    print_success "Unit tests completed"
}

run_integration_tests() {
    print_section "Running Integration Tests"

    print_info "Running integration tests..."
    if ! run_python_cmd "pytest tests/integration/ -v"; then
        # pytest exits with 5 if no tests are found
        if [ $? -eq 5 ]; then
            print_warning "No integration tests found"
        else
            print_error "Integration tests failed"
            exit 1
        fi
    fi
    print_success "Integration tests passed"
}

run_e2e_tests() {
    print_section "Running End-to-End Tests"

    print_info "Running e2e tests..."
    run_python_cmd "pytest tests/e2e/ -v"
    exit_code=$?
    # pytest exits with 5 if no tests are found
    if [ $exit_code -eq 5 ]; then
        print_warning "No e2e tests found"
    elif [ $exit_code -ne 0 ]; then
        print_error "End-to-end tests failed"
        exit 1
    else
        print_success "E2E tests passed"
    fi
}

# Linting and formatting
lint_all() {
    print_section "Linting Code"

    print_info "Running ruff linter..."
    if ! run_python_cmd "ruff check src/nocp tests examples"; then
        print_error "Ruff linting failed"
        exit 1
    fi
    print_success "Ruff linting passed"

    print_info "Running mypy type checker..."
    if ! run_python_cmd "mypy src/nocp --ignore-missing-imports"; then
        print_warning "Mypy type checking completed with warnings"
    else
        print_success "Mypy type checking passed"
    fi
}

format_all() {
    print_section "Formatting Code"

    print_info "Formatting Python code with ruff..."
    run_python_cmd "ruff format src/nocp tests examples"
    print_success "Code formatted"

    # Format configuration files if prettier is available
    if command_exists npx || command_exists prettier; then
        print_info "Formatting configuration files..."
        if command_exists npx; then
            npx prettier --write "**/*.md" "**/*.json" "**/*.yml" "**/*.yaml" "**/*.toml" 2>/dev/null || true
        elif command_exists prettier; then
            prettier --write "**/*.md" "**/*.json" "**/*.yml" "**/*.yaml" "**/*.toml" 2>/dev/null || true
        fi
        print_success "Configuration files formatted"
    fi
}

# Development and examples
run_example() {
    local example="${1:-basic_usage}"
    print_section "Running Example: $example"

    if [ ! -f "examples/${example}.py" ]; then
        print_error "Example not found: examples/${example}.py"
        list_examples
        return 1
    fi

    print_info "Running examples/${example}.py..."
    run_python_cmd "python examples/${example}.py"
}

list_examples() {
    print_section "Available Examples"

    if [ -d "examples" ]; then
        echo -e "${BLUE}Examples:${NC}"
        for example in examples/*.py; do
            if [ -f "$example" ]; then
                local name=$(basename "$example" .py)
                echo -e "  ${CYAN}• $name${NC}"
            fi
        done
        echo ""
        echo -e "${YELLOW}Run with: ./build.sh example <name>${NC}"
    else
        print_warning "No examples directory found"
    fi
}

# Benchmarking
run_benchmarks() {
    print_section "Running Benchmarks"

    if [ ! -d "benchmarks" ]; then
        print_warning "No benchmarks directory found"
        return 0
    fi

    print_info "Running performance benchmarks..."
    run_python_cmd "python -m pytest benchmarks/ -v --benchmark-only"
    pytest_exit_code=$?
    # pytest exits with 5 if no tests are found
    if [ $pytest_exit_code -ne 0 ]; then
        if [ $pytest_exit_code -eq 5 ]; then
            print_warning "No benchmarks found"
        else
            print_error "Benchmarks failed"
            exit 1
        fi
    else
        print_success "Benchmarks completed"
    fi
}

# Documentation
build_docs() {
    print_section "Building Documentation"

    print_info "Generating API documentation..."
    if run_python_cmd "python -m pdoc src/nocp --output-dir docs/api" 2>/dev/null; then
        print_success "Documentation generated in docs/api/"
    else
        print_warning "Documentation generation not available (install pdoc: uv pip install pdoc)"
    fi
}

# Cleaning
clean_all() {
    print_section "Cleaning All Build Artifacts"

    print_info "Cleaning Python artifacts..."
    rm -rf build dist *.egg-info
    rm -rf .pytest_cache .coverage htmlcov
    rm -rf .mypy_cache .ruff_cache
    find . -type d -name "__pycache__" -delete 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_success "Python artifacts cleaned"

    print_info "Cleaning documentation artifacts..."
    rm -rf docs/api
    print_success "Documentation artifacts cleaned"

    print_success "All artifacts cleaned"
}

clean_cache() {
    print_section "Cleaning Caches"

    print_info "Cleaning Python caches..."
    rm -rf .pytest_cache .mypy_cache .ruff_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_success "Caches cleaned"
}

# Help/Usage
show_help() {
    print_header
    echo -e "\n${BLUE}Usage: $0 [command]${NC}\n"

    echo -e "${YELLOW}Setup Commands:${NC}"
    echo -e "  ${GREEN}setup${NC}, ${GREEN}deps${NC}       Install all dependencies (auto-detects uv/poetry/pip)"
    echo -e "  ${GREEN}setup-litellm${NC}     Install with optional LiteLLM support"
    echo ""

    echo -e "${YELLOW}Build Commands:${NC}"
    echo -e "  ${GREEN}build${NC}             Build and validate NOCP (type checks + linting)"
    echo -e "  ${GREEN}lint${NC}              Run linting (ruff + mypy)"
    echo -e "  ${GREEN}format${NC}            Format code (ruff format)"
    echo ""

    echo -e "${YELLOW}Testing Commands:${NC}"
    echo -e "  ${GREEN}test${NC}              Run all tests with coverage"
    echo -e "  ${GREEN}test-unit${NC}         Run unit tests only"
    echo -e "  ${GREEN}test-integration${NC}  Run integration tests only"
    echo -e "  ${GREEN}test-e2e${NC}          Run end-to-end tests only"
    echo -e "  ${GREEN}benchmark${NC}         Run performance benchmarks"
    echo ""

    echo -e "${YELLOW}Development Commands:${NC}"
    echo -e "  ${GREEN}example [name]${NC}    Run an example (default: basic_usage)"
    echo -e "  ${GREEN}examples${NC}          List available examples"
    echo -e "  ${GREEN}docs${NC}              Generate API documentation"
    echo ""

    echo -e "${YELLOW}Utility Commands:${NC}"
    echo -e "  ${GREEN}status${NC}            Show detailed git and GitHub repository status"
    echo -e "  ${GREEN}clean${NC}             Remove all build artifacts and caches"
    echo -e "  ${GREEN}clean-cache${NC}       Remove only cache directories"
    echo -e "  ${GREEN}help${NC}              Show this help message"
    echo ""

    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ${CYAN}$0 setup && $0 build${NC}              # Full setup and build"
    echo -e "  ${CYAN}$0 test${NC}                           # Run tests with coverage"
    echo -e "  ${CYAN}$0 format && $0 lint && $0 test${NC}   # Full QA pipeline"
    echo -e "  ${CYAN}$0 example basic_usage${NC}            # Run basic usage example"
    echo -e "  ${CYAN}$0 benchmark${NC}                      # Run performance benchmarks"
    echo -e "  ${CYAN}$0 status${NC}                         # Check git/GitHub status"
    echo ""

    echo -e "${YELLOW}Environment Variables:${NC}"
    echo -e "  ${CYAN}INSTALL_LITELLM=true${NC}              Install with LiteLLM support"
    echo -e "  ${CYAN}GEMINI_API_KEY=<key>${NC}              Set Gemini API key for examples"
    echo ""

    echo -e "${YELLOW}Detected Tools:${NC}"
    echo -e "  Python Manager: ${CYAN}$(detect_python_manager)${NC}"
    echo -e "  GitHub CLI: ${CYAN}$(command_exists gh && echo "available (enhanced status)" || echo "not found")${NC}"
    echo -e "  JSON Parser: ${CYAN}$(command_exists jq && echo "available" || echo "not found (limited GitHub features)")${NC}"
    echo ""
}

# Main command dispatcher
case "${1:-help}" in
    "setup"|"deps"|"install")
        print_header
        install_dependencies
        print_success "Setup complete! Run '$0 build' to validate the installation"
        ;;

    "setup-litellm")
        print_header
        INSTALL_LITELLM=true install_dependencies
        print_success "Setup complete with LiteLLM!"
        ;;

    "build")
        print_header
        build_all
        ;;

    "test")
        print_header
        run_tests
        ;;

    "test-unit")
        print_header
        run_unit_tests
        ;;

    "test-integration")
        print_header
        run_integration_tests
        ;;

    "test-e2e")
        print_header
        run_e2e_tests
        ;;

    "lint")
        print_header
        lint_all
        ;;

    "format")
        print_header
        format_all
        ;;

    "example")
        print_header
        run_example "${2:-basic_usage}"
        ;;

    "examples"|"list-examples")
        print_header
        list_examples
        ;;

    "benchmark"|"benchmarks")
        print_header
        run_benchmarks
        ;;

    "docs"|"documentation")
        print_header
        build_docs
        ;;

    "status")
        print_header
        show_git_status
        ;;

    "clean")
        print_header
        clean_all
        ;;

    "clean-cache")
        print_header
        clean_cache
        ;;

    "help"|"-h"|"--help")
        show_help
        ;;

    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
