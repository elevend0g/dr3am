"""
dr3am - Autonomous Agent MCP Server

Transform reactive chatbots into proactive research assistants.
"""

__version__ = "1.0.0"
__author__ = "dr3am Team"
__email__ = "team@dr3am.ai"

from .core.semantic_analyzer import SemanticInterestAnalyzer, ConversationMessage, DetectedInterest
from .core.research_engine import ResearchOrchestrator, ResearchResult, ActionableInsight
from .core.mcp_server import Dr3amMCPServer

__all__ = [
    "SemanticInterestAnalyzer",
    "ConversationMessage", 
    "DetectedInterest",
    "ResearchOrchestrator",
    "ResearchResult",
    "ActionableInsight",
    "Dr3amMCPServer",
]