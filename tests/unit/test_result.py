"""Unit tests for Result pattern implementation"""
import pytest
from nocp.models.result import Result


class TestResultCreation:
    """Test Result creation methods"""

    def test_ok_creates_successful_result(self):
        """Test creating successful result"""
        result = Result.ok(42)

        assert result.success is True
        assert result.data == 42
        assert result.error is None
        assert result.warnings == []

    def test_err_creates_failed_result(self):
        """Test creating failed result"""
        result = Result.err("Something went wrong")

        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"
        assert result.warnings == []

    def test_ok_with_complex_data(self):
        """Test Result.ok with complex data types"""
        data = {"user": "Alice", "age": 30, "items": [1, 2, 3]}
        result = Result.ok(data)

        assert result.success is True
        assert result.data == data
        assert result.data["user"] == "Alice"

    def test_ok_with_none_raises_error(self):
        """Test that Result.ok with None raises ValueError"""
        with pytest.raises(ValueError, match="Successful result must have data"):
            Result(success=True, data=None)

    def test_err_without_error_raises_error(self):
        """Test that Result.err without error raises ValueError"""
        with pytest.raises(ValueError, match="Failed result must have error"):
            Result(success=False, error=None)


class TestResultUnwrapping:
    """Test Result unwrapping methods"""

    def test_unwrap_successful_result(self):
        """Test unwrapping successful result"""
        result = Result.ok(42)
        assert result.unwrap() == 42

    def test_unwrap_failed_result_raises(self):
        """Test unwrapping failed result raises ValueError"""
        result = Result.err("Failed operation")

        with pytest.raises(ValueError, match="Unwrap called on failed result"):
            result.unwrap()

    def test_unwrap_or_with_successful_result(self):
        """Test unwrap_or with successful result returns data"""
        result = Result.ok(42)
        assert result.unwrap_or(0) == 42

    def test_unwrap_or_with_failed_result(self):
        """Test unwrap_or with failed result returns default"""
        result = Result.err("Failed")
        assert result.unwrap_or(0) == 0

    def test_unwrap_or_else_with_successful_result(self):
        """Test unwrap_or_else with successful result returns data"""
        result = Result.ok(42)
        assert result.unwrap_or_else(lambda: 0) == 42

    def test_unwrap_or_else_with_failed_result(self):
        """Test unwrap_or_else with failed result calls function"""
        result = Result.err("Failed")
        assert result.unwrap_or_else(lambda: 100) == 100

    def test_expect_successful_result(self):
        """Test expect with successful result returns data"""
        result = Result.ok(42)
        assert result.expect("Should not fail") == 42

    def test_expect_failed_result_raises_with_custom_message(self):
        """Test expect with failed result raises with custom message"""
        result = Result.err("Connection timeout")

        with pytest.raises(ValueError, match="Custom error: Connection timeout"):
            result.expect("Custom error")


class TestResultMapping:
    """Test Result mapping and transformation"""

    def test_map_successful_result(self):
        """Test mapping function over successful result"""
        result = Result.ok(5)
        mapped = result.map(lambda x: x * 2)

        assert mapped.success is True
        assert mapped.data == 10

    def test_map_failed_result(self):
        """Test mapping over failed result preserves error"""
        result = Result.err("Error")
        mapped = result.map(lambda x: x * 2)

        assert mapped.success is False
        assert mapped.error == "Error"

    def test_map_chain(self):
        """Test chaining multiple map operations"""
        result = Result.ok(5) \
            .map(lambda x: x * 2) \
            .map(lambda x: x + 3) \
            .map(lambda x: str(x))

        assert result.success is True
        assert result.data == "13"

    def test_map_with_exception(self):
        """Test map that raises exception"""
        result = Result.ok(5)
        mapped = result.map(lambda x: x / 0)

        assert mapped.success is False
        assert "Map failed" in mapped.error
        assert "division by zero" in mapped.error.lower()

    def test_map_preserves_warnings(self):
        """Test that map preserves warnings"""
        result = Result.ok(5)
        result.add_warning("Warning 1")

        mapped = result.map(lambda x: x * 2)

        assert mapped.success is True
        assert mapped.data == 10
        assert "Warning 1" in mapped.warnings


class TestResultChaining:
    """Test Result chaining with and_then"""

    def test_and_then_successful(self):
        """Test and_then with successful results"""
        def divide_by_two(x: int) -> Result[float]:
            if x % 2 == 0:
                return Result.ok(x / 2)
            return Result.err("Not divisible by 2")

        result = Result.ok(10).and_then(divide_by_two)

        assert result.success is True
        assert result.data == 5.0

    def test_and_then_with_failure(self):
        """Test and_then when inner function fails"""
        def divide_by_two(x: int) -> Result[float]:
            if x % 2 == 0:
                return Result.ok(x / 2)
            return Result.err("Not divisible by 2")

        result = Result.ok(5).and_then(divide_by_two)

        assert result.success is False
        assert result.error == "Not divisible by 2"

    def test_and_then_with_initial_failure(self):
        """Test and_then with initially failed result"""
        def process(x: int) -> Result[int]:
            return Result.ok(x * 2)

        result = Result.err("Initial error").and_then(process)

        assert result.success is False
        assert result.error == "Initial error"

    def test_and_then_chain(self):
        """Test chaining multiple and_then operations"""
        def add_five(x: int) -> Result[int]:
            return Result.ok(x + 5)

        def multiply_by_two(x: int) -> Result[int]:
            return Result.ok(x * 2)

        def to_string(x: int) -> Result[str]:
            return Result.ok(str(x))

        result = Result.ok(3) \
            .and_then(add_five) \
            .and_then(multiply_by_two) \
            .and_then(to_string)

        assert result.success is True
        assert result.data == "16"  # (3 + 5) * 2 = 16

    def test_and_then_with_exception(self):
        """Test and_then when function raises exception"""
        def raises_error(x: int) -> Result[int]:
            raise RuntimeError("Unexpected error")

        result = Result.ok(5).and_then(raises_error)

        assert result.success is False
        assert "and_then failed" in result.error


class TestResultFallback:
    """Test Result fallback with or_else"""

    def test_or_else_with_successful_result(self):
        """Test or_else with successful result returns original"""
        result = Result.ok(42)
        fallback = result.or_else(lambda e: Result.ok(0))

        assert fallback.success is True
        assert fallback.data == 42

    def test_or_else_with_failed_result(self):
        """Test or_else with failed result calls fallback"""
        result = Result.err("Primary failed")
        fallback = result.or_else(lambda e: Result.ok(100))

        assert fallback.success is True
        assert fallback.data == 100

    def test_or_else_receives_error(self):
        """Test that or_else receives the error message"""
        received_error = None

        def capture_error(error: str) -> Result[int]:
            nonlocal received_error
            received_error = error
            return Result.ok(0)

        Result.err("Test error").or_else(capture_error)

        assert received_error == "Test error"

    def test_or_else_can_fail(self):
        """Test or_else fallback can also fail"""
        result = Result.err("Primary failed")
        fallback = result.or_else(lambda e: Result.err("Fallback also failed"))

        assert fallback.success is False
        assert fallback.error == "Fallback also failed"


class TestResultHelpers:
    """Test Result helper methods"""

    def test_is_ok_with_successful_result(self):
        """Test is_ok returns True for successful result"""
        result = Result.ok(42)
        assert result.is_ok() is True
        assert result.is_err() is False

    def test_is_err_with_failed_result(self):
        """Test is_err returns True for failed result"""
        result = Result.err("Failed")
        assert result.is_ok() is False
        assert result.is_err() is True

    def test_from_optional_with_value(self):
        """Test from_optional with non-None value"""
        result = Result.from_optional(42)

        assert result.success is True
        assert result.data == 42

    def test_from_optional_with_none(self):
        """Test from_optional with None"""
        result = Result.from_optional(None)

        assert result.success is False
        assert result.error == "Value is None"

    def test_from_optional_with_custom_error(self):
        """Test from_optional with custom error message"""
        result = Result.from_optional(None, "Custom error message")

        assert result.success is False
        assert result.error == "Custom error message"

    def test_from_exception_with_success(self):
        """Test from_exception with successful function"""
        result = Result.from_exception(lambda: 42)

        assert result.success is True
        assert result.data == 42

    def test_from_exception_with_failure(self):
        """Test from_exception with failing function"""
        def raises_error():
            raise ValueError("Test error")

        result = Result.from_exception(raises_error)

        assert result.success is False
        assert "Operation failed" in result.error
        assert "Test error" in result.error

    def test_from_exception_with_custom_prefix(self):
        """Test from_exception with custom error prefix"""
        def raises_error():
            raise ValueError("Test error")

        result = Result.from_exception(raises_error, error_prefix="Custom prefix")

        assert result.success is False
        assert "Custom prefix" in result.error


class TestResultWarnings:
    """Test Result warning management"""

    def test_add_warning(self):
        """Test adding warning to result"""
        result = Result.ok(42)
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        assert len(result.warnings) == 2
        assert "Warning 1" in result.warnings
        assert "Warning 2" in result.warnings

    def test_add_warning_returns_self(self):
        """Test add_warning returns self for chaining"""
        result = Result.ok(42)
        returned = result.add_warning("Warning")

        assert returned is result

    def test_warnings_in_failed_result(self):
        """Test warnings in failed result"""
        result = Result.err("Error")
        result.add_warning("Warning before failure")

        assert result.success is False
        assert len(result.warnings) == 1


class TestResultRepresentation:
    """Test Result string representation"""

    def test_repr_successful_result(self):
        """Test repr of successful result"""
        result = Result.ok(42)
        assert repr(result) == "Result.ok(42)"

    def test_repr_failed_result(self):
        """Test repr of failed result"""
        result = Result.err("Error message")
        assert repr(result) == "Result.err('Error message')"

    def test_repr_with_warnings(self):
        """Test repr includes warnings"""
        result = Result.ok(42)
        result.add_warning("Test warning")

        assert "warnings=" in repr(result)
        assert "Test warning" in repr(result)

    def test_bool_conversion_successful(self):
        """Test Result can be used in boolean context"""
        result = Result.ok(42)
        assert bool(result) is True
        assert result  # Direct boolean check

    def test_bool_conversion_failed(self):
        """Test failed Result evaluates to False"""
        result = Result.err("Error")
        assert bool(result) is False
        assert not result  # Direct boolean check


class TestResultRealWorldScenarios:
    """Test Result in realistic scenarios"""

    def test_database_query_simulation(self):
        """Simulate database query with Result"""
        def query_user(user_id: int) -> Result[dict]:
            if user_id <= 0:
                return Result.err("Invalid user ID")

            # Simulate database lookup
            if user_id == 1:
                return Result.ok({"id": 1, "name": "Alice"})
            return Result.err("User not found")

        # Successful query
        result = query_user(1)
        assert result.success is True
        assert result.data["name"] == "Alice"

        # Failed query
        result = query_user(999)
        assert result.success is False
        assert "not found" in result.error

    def test_validation_pipeline(self):
        """Test validation pipeline with Result"""
        def validate_age(age: int) -> Result[int]:
            if age < 0:
                return Result.err("Age cannot be negative")
            if age > 150:
                return Result.err("Age too high")
            return Result.ok(age)

        def validate_name(name: str) -> Result[str]:
            if not name:
                return Result.err("Name cannot be empty")
            if len(name) < 2:
                return Result.err("Name too short")
            return Result.ok(name)

        # Valid inputs
        age_result = validate_age(25)
        name_result = validate_name("Alice")

        assert age_result.success is True
        assert name_result.success is True

        # Invalid inputs
        age_result = validate_age(-5)
        name_result = validate_name("")

        assert age_result.success is False
        assert name_result.success is False

    def test_file_processing_pipeline(self):
        """Simulate file processing with Result chain"""
        def read_file(path: str) -> Result[str]:
            if not path:
                return Result.err("Path cannot be empty")
            # Simulate reading
            return Result.ok("file contents")

        def parse_json(content: str) -> Result[dict]:
            if not content:
                return Result.err("Empty content")
            # Simulate parsing
            return Result.ok({"data": "parsed"})

        def validate_schema(data: dict) -> Result[dict]:
            if "data" not in data:
                return Result.err("Missing 'data' field")
            return Result.ok(data)

        # Successful pipeline
        result = read_file("test.json") \
            .and_then(parse_json) \
            .and_then(validate_schema)

        assert result.success is True
        assert result.data == {"data": "parsed"}

        # Failed pipeline (early failure)
        result = read_file("") \
            .and_then(parse_json) \
            .and_then(validate_schema)

        assert result.success is False
        assert "Path cannot be empty" in result.error

    def test_error_recovery_with_fallback(self):
        """Test error recovery using or_else"""
        def primary_operation(x: int) -> Result[int]:
            if x < 0:
                return Result.err("Negative input")
            return Result.ok(x * 2)

        def fallback_operation(error: str) -> Result[int]:
            # Provide default value on error
            return Result.ok(0)

        # Primary succeeds
        result = primary_operation(5).or_else(fallback_operation)
        assert result.data == 10

        # Primary fails, fallback succeeds
        result = primary_operation(-5).or_else(fallback_operation)
        assert result.data == 0
