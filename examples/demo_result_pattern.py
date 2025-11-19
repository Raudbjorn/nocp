"""
Example: Using Result Pattern for Error Handling in NOCP

This demonstrates how to use the Result type for explicit error handling
in tool execution, context management, and other operations.
"""

from nocp.models.result import Result
from nocp.models.schemas import ToolDefinition, ToolParameter
from nocp.modules.tool_executor import ToolExecutor


# Example 1: Tool Execution with Result Pattern
# ==============================================

def search_database(query: str, limit: int = 10) -> Result[dict]:
    """
    Simulate a database search that can fail.

    Returns:
        Result containing search results or error message
    """
    if not query:
        return Result.err("Query cannot be empty")

    if limit <= 0:
        return Result.err("Limit must be positive")

    # Simulate database search
    # In real scenario, this would query a database
    results = {
        "query": query,
        "results": [
            {"id": 1, "title": "Result 1", "score": 0.95},
            {"id": 2, "title": "Result 2", "score": 0.87},
        ][:limit],
        "total_count": 2,
    }

    return Result.ok(results)


def validate_and_execute_search(query: str, limit: int) -> Result[dict]:
    """
    Validate inputs and execute search with Result pattern.

    This shows how to chain validations and operations.
    """
    # Validate query
    if not query or len(query) < 3:
        return Result.err("Query must be at least 3 characters")

    # Validate limit
    if limit < 1 or limit > 100:
        return Result.err("Limit must be between 1 and 100")

    # Execute search
    return search_database(query, limit)


# Example 2: Chaining Operations with Result
# ===========================================

def parse_query(raw_query: str) -> Result[dict]:
    """Parse raw query into structured format"""
    if not raw_query:
        return Result.err("Empty query")

    # Simple parsing
    parts = raw_query.split(":")
    if len(parts) != 2:
        return Result.err("Invalid query format. Expected 'field:value'")

    field, value = parts
    return Result.ok({"field": field.strip(), "value": value.strip()})


def validate_query(parsed_query: dict) -> Result[dict]:
    """Validate parsed query structure"""
    allowed_fields = {"name", "email", "status"}

    field = parsed_query.get("field")
    if field not in allowed_fields:
        return Result.err(f"Invalid field '{field}'. Allowed: {allowed_fields}")

    return Result.ok(parsed_query)


def execute_query(query: dict) -> Result[list]:
    """Execute the validated query"""
    # Simulate database query
    results = [
        {"id": 1, f"{query['field']}": query['value']},
        {"id": 2, f"{query['field']}": query['value']},
    ]
    return Result.ok(results)


def query_pipeline(raw_query: str) -> Result[list]:
    """
    Complete query pipeline using Result chaining.

    This shows the power of and_then for error propagation.
    """
    return (
        parse_query(raw_query)
        .and_then(validate_query)
        .and_then(execute_query)
    )


# Example 3: Error Recovery with Fallbacks
# =========================================

def fetch_from_primary_db(key: str) -> Result[dict]:
    """Fetch data from primary database"""
    # Simulate primary DB failure
    return Result.err("Primary database unavailable")


def fetch_from_cache(key: str) -> Result[dict]:
    """Fetch data from cache as fallback"""
    # Simulate cache hit
    return Result.ok({"key": key, "value": "cached_data", "source": "cache"})


def fetch_with_fallback(key: str) -> Result[dict]:
    """
    Fetch data with automatic fallback to cache.

    This shows how or_else enables graceful degradation.
    """
    return fetch_from_primary_db(key).or_else(
        lambda error: fetch_from_cache(key).add_warning(f"Primary DB failed: {error}")
    )


# Example 4: Tool Executor Integration
# =====================================

def create_safe_tool_wrapper(tool_name: str, tool_func: callable) -> callable:
    """
    Wrap a tool function to return Result instead of raising exceptions.

    This shows how to integrate Result pattern with existing tools.
    """
    def wrapper(**kwargs) -> Result[dict]:
        try:
            # Execute the original tool
            result = tool_func(**kwargs)

            # Wrap successful result
            return Result.ok({
                "tool_name": tool_name,
                "output": result,
                "success": True,
            })

        except ValueError as e:
            # Handle validation errors
            return Result.err(f"Validation error in {tool_name}: {str(e)}")

        except Exception as e:
            # Handle unexpected errors
            return Result.err(f"Unexpected error in {tool_name}: {str(e)}")

    return wrapper


# Example 5: Batch Operations with Result
# ========================================

def process_items_batch(items: list) -> Result[dict]:
    """
    Process multiple items, collecting successes and failures.

    This shows how to handle partial failures in batch operations.
    """
    if not items:
        return Result.err("No items to process")

    successes = []
    failures = []

    for item in items:
        # Process each item
        result = process_single_item(item)

        if result.success:
            successes.append(result.data)
        else:
            failures.append({"item": item, "error": result.error})

    # Create result with warnings for failures
    batch_result = Result.ok({
        "processed": len(successes),
        "failed": len(failures),
        "successes": successes,
        "failures": failures,
    })

    # Add warnings for each failure
    for failure in failures:
        batch_result.add_warning(f"Failed to process {failure['item']}: {failure['error']}")

    return batch_result


def process_single_item(item: str) -> Result[dict]:
    """Process a single item"""
    if not item:
        return Result.err("Empty item")

    # Simulate processing
    return Result.ok({"item": item, "processed": True})


# Example 6: Configuration Validation
# ====================================

def validate_config(config: dict) -> Result[dict]:
    """
    Validate configuration with detailed error messages.

    This shows how Result enables clear validation chains.
    """
    # Check required fields
    required_fields = ["api_key", "model", "max_tokens"]
    missing_fields = [f for f in required_fields if f not in config]

    if missing_fields:
        return Result.err(f"Missing required fields: {', '.join(missing_fields)}")

    # Validate max_tokens
    max_tokens = config.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        return Result.err("max_tokens must be a positive integer")

    # Validate model
    valid_models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
    if config["model"] not in valid_models:
        return Result.err(f"Invalid model. Must be one of: {', '.join(valid_models)}")

    return Result.ok(config)


# Example 7: Practical Usage Patterns
# ====================================

def main():
    """Demonstrate various Result usage patterns"""

    print("=" * 60)
    print("NOCP Result Pattern Examples")
    print("=" * 60)

    # Example 1: Basic success/failure handling
    print("\n1. Basic Search:")
    result = search_database("test query", limit=5)
    if result.success:
        print(f"   ✓ Found {result.data['total_count']} results")
    else:
        print(f"   ✗ Error: {result.error}")

    # Example 2: Chained operations
    print("\n2. Query Pipeline:")
    result = query_pipeline("name:Alice")
    if result.success:
        print(f"   ✓ Query executed, found {len(result.data)} results")
    else:
        print(f"   ✗ Pipeline failed: {result.error}")

    result = query_pipeline("invalid_query")
    if result.success:
        print(f"   ✓ Query executed")
    else:
        print(f"   ✗ Pipeline failed: {result.error}")

    # Example 3: Fallback handling
    print("\n3. Fallback to Cache:")
    result = fetch_with_fallback("user:123")
    if result.success:
        print(f"   ✓ Data fetched from: {result.data['source']}")
        if result.warnings:
            print(f"   ⚠ Warnings: {', '.join(result.warnings)}")
    else:
        print(f"   ✗ Error: {result.error}")

    # Example 4: Batch processing
    print("\n4. Batch Processing:")
    items = ["item1", "item2", "", "item3"]
    result = process_items_batch(items)
    if result.success:
        data = result.data
        print(f"   ✓ Processed: {data['processed']}, Failed: {data['failed']}")
        if result.warnings:
            print(f"   ⚠ Warnings:")
            for warning in result.warnings:
                print(f"     - {warning}")
    else:
        print(f"   ✗ Error: {result.error}")

    # Example 5: Configuration validation
    print("\n5. Config Validation:")
    config = {
        "api_key": "sk-test123",
        "model": "gpt-4",
        "max_tokens": 4096,
    }
    result = validate_config(config)
    if result.success:
        print(f"   ✓ Config valid")
    else:
        print(f"   ✗ Invalid config: {result.error}")

    invalid_config = {"api_key": "test"}
    result = validate_config(invalid_config)
    if result.success:
        print(f"   ✓ Config valid")
    else:
        print(f"   ✗ Invalid config: {result.error}")

    # Example 6: Unwrapping patterns
    print("\n6. Unwrapping Patterns:")

    # Safe unwrapping with default
    result = Result.err("Something failed")
    value = result.unwrap_or({"default": "value"})
    print(f"   Got value (with default): {value}")

    # Map transformation
    result = Result.ok(5)
    squared = result.map(lambda x: x ** 2)
    print(f"   Mapped result: {squared.data}")

    # Chained map
    result = Result.ok(10).map(lambda x: x * 2).map(lambda x: x + 5)
    print(f"   Chained map: {result.data}")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
