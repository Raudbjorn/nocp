"""
NOCP - High-Efficiency LLM Proxy Agent
Token-Oriented Optimization Layer for Large Context Models
"""

from .core.agent import HighEfficiencyProxyAgent
from .models.schemas import AgentRequest, AgentResponse, ToolDefinition
from .core.config import ProxyConfig
from .utils.rich_logging import setup_rich_logging

__version__ = "0.1.0"

__all__ = [
    "HighEfficiencyProxyAgent",
    "AgentRequest",
    "AgentResponse",
    "ToolDefinition",
    "ProxyConfig",
    "setup_rich_logging",  # Export for users who want rich tracebacks
]
