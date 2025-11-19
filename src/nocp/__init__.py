"""
NOCP - High-Efficiency LLM Proxy Agent
Token-Oriented Optimization Layer for Large Context Models
"""

# Setup rich logging and tracebacks globally
from .utils.rich_logging import setup_rich_logging
setup_rich_logging()

from .core.agent import HighEfficiencyProxyAgent
from .models.schemas import AgentRequest, AgentResponse, ToolDefinition
from .core.config import ProxyConfig

__version__ = "0.1.0"

__all__ = [
    "HighEfficiencyProxyAgent",
    "AgentRequest",
    "AgentResponse",
    "ToolDefinition",
    "ProxyConfig",
]
