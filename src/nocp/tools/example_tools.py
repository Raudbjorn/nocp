"""
Example tool implementations for demonstration purposes.

These tools simulate common scenarios where context compression is beneficial:
- Database/RAG searches (semantic pruning target)
- Verbose API responses (knowledge distillation target)
- Document retrieval (semantic pruning target)
"""

from typing import Callable
from ..models.schemas import ToolDefinition, ToolParameter


def search_database(query: str, limit: int = 10) -> str:
    """
    Simulated database search tool.

    Returns verbose results that benefit from semantic pruning.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        Verbose search results
    """
    # Simulate verbose database results
    results = []
    for i in range(min(limit, 5)):
        results.append(f"""
        Result {i+1}:
        Title: Product {i+1} - {query} Edition
        Description: This is a detailed description of product {i+1} that matches your search query '{query}'.
        It includes extensive information about features, specifications, pricing, and availability.

        Features:
        - High-quality materials and construction
        - Advanced technology integration
        - User-friendly interface
        - Comprehensive warranty coverage
        - Energy-efficient operation
        - Sustainable manufacturing processes

        Specifications:
        - Model: PRD-{i+1}-2024
        - SKU: {query.upper()}-{i+1:03d}
        - Dimensions: 10" x 8" x 6"
        - Weight: 2.5 lbs
        - Color options: Black, White, Silver, Blue
        - Material: Premium aluminum alloy

        Pricing:
        - List Price: ${49.99 + i*10}
        - Sale Price: ${39.99 + i*10}
        - Bulk Discount: Available for 10+ units

        Availability: In stock - Ships within 24 hours
        Customer Rating: 4.{5+i}/5.0 based on {100+i*50} reviews
        """)

    return "\n\n".join(results)


def analyze_data(dataset: str, metrics: str = "all") -> str:
    """
    Simulated data analysis tool.

    Returns verbose analytical results that benefit from knowledge distillation.

    Args:
        dataset: Dataset identifier
        metrics: Metrics to analyze

    Returns:
        Verbose analysis results
    """
    # Simulate verbose analytical output
    return f"""
    DATA ANALYSIS REPORT
    ====================
    Dataset: {dataset}
    Analysis Date: 2024-01-15
    Metrics Requested: {metrics}

    SUMMARY STATISTICS:
    -------------------
    Total Records: 10,542
    Valid Records: 10,489 (99.5%)
    Invalid Records: 53 (0.5%)
    Date Range: 2023-01-01 to 2024-01-15

    DETAILED METRICS:
    -----------------

    1. Revenue Metrics:
       - Total Revenue: $1,245,678.90
       - Average Transaction: $118.35
       - Median Transaction: $89.50
       - Standard Deviation: $45.67
       - Minimum: $5.00
       - Maximum: $999.99
       - Growth Rate (YoY): +15.3%

    2. Customer Metrics:
       - Total Unique Customers: 3,456
       - New Customers: 892 (25.8%)
       - Returning Customers: 2,564 (74.2%)
       - Average Customer Lifetime Value: $360.42
       - Customer Retention Rate: 78.5%
       - Churn Rate: 21.5%

    3. Product Performance:
       - Top Selling Product: Widget Pro (1,234 units)
       - Lowest Selling Product: Gadget Basic (45 units)
       - Average Units per Transaction: 2.3
       - Product Return Rate: 3.2%
       - Average Product Rating: 4.2/5.0

    4. Temporal Patterns:
       - Peak Sales Day: Friday
       - Peak Sales Hour: 2:00 PM - 3:00 PM
       - Lowest Sales Day: Sunday
       - Seasonal Trend: Q4 strongest (35% of annual revenue)

    5. Geographic Distribution:
       - Top Region: Northeast (42%)
       - Second: West Coast (28%)
       - Third: Midwest (18%)
       - Fourth: Southeast (12%)

    RECOMMENDATIONS:
    ----------------
    1. Increase marketing efforts for low-performing products
    2. Optimize inventory for peak sales periods
    3. Develop retention programs for at-risk customers
    4. Expand operations in high-performing regions
    5. Investigate causes of product returns

    TECHNICAL DETAILS:
    ------------------
    Analysis Method: Statistical regression with confidence intervals
    Confidence Level: 95%
    Error Margin: ±2.5%
    Data Quality Score: 9.2/10
    Processing Time: 2.34 seconds
    """


def fetch_document(doc_id: str) -> str:
    """
    Simulated document retrieval tool.

    Returns verbose document content suitable for semantic pruning.

    Args:
        doc_id: Document identifier

    Returns:
        Verbose document content
    """
    return f"""
    DOCUMENT RETRIEVAL SYSTEM
    =========================
    Document ID: {doc_id}
    Retrieved: 2024-01-15 10:30:00 UTC

    DOCUMENT METADATA:
    ------------------
    Title: Comprehensive Guide to {doc_id}
    Author: Technical Documentation Team
    Version: 3.2.1
    Last Updated: 2024-01-10
    Page Count: 47
    Word Count: 12,456
    Format: PDF
    Size: 2.3 MB
    Classification: Public

    DOCUMENT CONTENT:
    -----------------

    CHAPTER 1: INTRODUCTION

    This comprehensive guide provides detailed information about {doc_id} and its
    implementation in modern systems. The document covers theoretical foundations,
    practical applications, best practices, and common troubleshooting scenarios.

    Section 1.1: Overview
    {doc_id} is a critical component in enterprise architecture that enables
    seamless integration between disparate systems. It provides a unified interface
    for data exchange, transformation, and routing across multiple platforms.

    Section 1.2: Key Benefits
    - Improved system interoperability
    - Reduced development time and costs
    - Enhanced scalability and performance
    - Simplified maintenance and updates
    - Comprehensive security features
    - Built-in monitoring and logging

    CHAPTER 2: ARCHITECTURE

    Section 2.1: System Components
    The architecture consists of several key components working together:

    1. Core Engine: Handles primary processing logic
    2. Data Layer: Manages persistence and caching
    3. API Gateway: Provides external interface
    4. Security Module: Enforces authentication and authorization
    5. Monitoring System: Tracks performance and health

    Section 2.2: Integration Points
    The system integrates with various external services and platforms:
    - RESTful APIs for synchronous communication
    - Message queues for asynchronous processing
    - Database connections for data persistence
    - Third-party services for extended functionality

    CHAPTER 3: IMPLEMENTATION

    Section 3.1: Prerequisites
    Before implementation, ensure the following requirements are met:
    - System requirements: 8GB RAM, 4 CPU cores, 50GB storage
    - Software dependencies: Python 3.10+, PostgreSQL 14+, Redis 7+
    - Network requirements: Outbound HTTPS access, inbound port 8080
    - Security certificates: Valid SSL/TLS certificates

    Section 3.2: Installation Steps
    Follow these steps for successful installation:
    1. Download the latest release package
    2. Extract files to installation directory
    3. Configure environment variables
    4. Initialize database schema
    5. Start the service
    6. Verify installation

    [Additional 40 pages of detailed technical documentation...]

    APPENDIX A: API REFERENCE
    APPENDIX B: TROUBLESHOOTING GUIDE
    APPENDIX C: PERFORMANCE TUNING
    APPENDIX D: SECURITY BEST PRACTICES
    """


def create_search_database_tool() -> tuple[ToolDefinition, Callable]:
    """
    Create the search_database tool with its definition.

    Returns:
        Tuple of (ToolDefinition, callable)
    """
    definition = ToolDefinition(
        name="search_database",
        description="Search the product database for items matching the query. Returns detailed product information.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query string to find products",
                required=True,
            ),
            ToolParameter(
                name="limit",
                type="number",
                description="Maximum number of results to return (default: 10)",
                required=False,
            ),
        ],
        compression_threshold=3000,  # Custom threshold for this tool
    )

    return definition, search_database


def create_analyze_data_tool() -> tuple[ToolDefinition, Callable]:
    """
    Create the analyze_data tool with its definition.

    Returns:
        Tuple of (ToolDefinition, callable)
    """
    definition = ToolDefinition(
        name="analyze_data",
        description="Analyze a dataset and return comprehensive metrics and insights.",
        parameters=[
            ToolParameter(
                name="dataset",
                type="string",
                description="Dataset identifier to analyze",
                required=True,
            ),
            ToolParameter(
                name="metrics",
                type="string",
                description="Specific metrics to analyze (default: 'all')",
                required=False,
            ),
        ],
        compression_threshold=4000,
    )

    return definition, analyze_data


def create_fetch_document_tool() -> tuple[ToolDefinition, Callable]:
    """
    Create the fetch_document tool with its definition.

    Returns:
        Tuple of (ToolDefinition, callable)
    """
    definition = ToolDefinition(
        name="fetch_document",
        description="Retrieve a document by its ID. Returns full document content and metadata.",
        parameters=[
            ToolParameter(
                name="doc_id",
                type="string",
                description="Document identifier to retrieve",
                required=True,
            ),
        ],
        compression_threshold=5000,
    )

    return definition, fetch_document
