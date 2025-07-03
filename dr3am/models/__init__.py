"""Data models for dr3am"""

from .conversation import ConversationMessage
from .interests import DetectedInterest, ResearchOpportunity
from .research import ResearchResult, ActionableInsight

__all__ = [
    "ConversationMessage",
    "DetectedInterest",
    "ResearchOpportunity", 
    "ResearchResult",
    "ActionableInsight",
]