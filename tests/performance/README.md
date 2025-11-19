# Performance Benchmarks

Performance and throughput tests.

## Purpose

Performance tests measure system performance characteristics:

- Compression speed and throughput
- Memory usage and efficiency
- Request latency
- Concurrent request handling

## Coverage

- `test_compression_speed.py` - Compression speed benchmarks
- `test_throughput.py` - Request throughput and latency
- `test_memory_usage.py` - Memory consumption and leaks

## Running Performance Tests

```bash
# Run all performance tests
pytest tests/performance/

# Run with markers
pytest -m performance

# Run with slow tests enabled (recommended for benchmarks)
pytest tests/performance/ --run-slow
```

## Markers

- `@pytest.mark.performance` - Marks test as performance benchmark
- `@pytest.mark.slow` - For tests taking >1 second

## Notes

- Performance tests should run on consistent hardware
- Results may vary based on system load
- Use for regression testing and optimization validation
- Consider running in isolation for accurate measurements
