"""Interest and research opportunity models"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
from enum import Enum


class InterestType(Enum):
    HOBBY = "hobby"
    PROBLEM = "problem" 
    LEARNING = "learning"
    HEALTH = "health"
    GOAL = "goal"
    PREFERENCE = "preference"
    CONCERN = "concern"


class EngagementLevel(Enum):
    CASUAL_MENTION = 1      # Single mention, no follow-up
    RECURRING_INTEREST = 2   # Multiple mentions across conversations
    ACTIVE_ENGAGEMENT = 3    # Questions, requests for help, action taken
    PERSISTENT_FOCUS = 4     # Dominant topic, high emotional investment


@dataclass
class DetectedInterest:
    """Represents a discovered user interest or pattern"""
    topic: str
    interest_type: InterestType
    engagement_level: EngagementLevel
    confidence_score: float  # 0.0 to 1.0
    first_mentioned: datetime
    last_mentioned: datetime
    mention_count: int
    key_phrases: List[str]
    context_summary: str
    research_potential: float  # 0.0 to 1.0 - how actionable this is
    related_topics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate interest data"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        if not 0.0 <= self.research_potential <= 1.0:
            raise ValueError("Research potential must be between 0.0 and 1.0")
        
        if self.mention_count < 1:
            raise ValueError("Mention count must be at least 1")
        
        if self.first_mentioned > self.last_mentioned:
            raise ValueError("First mentioned date cannot be after last mentioned date")


@dataclass
class ResearchOpportunity:
    """Actionable research opportunity derived from interests"""
    interest_topic: str
    research_type: str  # "find_resources", "deep_dive", "monitor", "comparison"
    priority_score: float  # 0.0 to 1.0
    research_query: str
    expected_value: str  # "high", "medium", "low"
    urgency: str  # "urgent", "high", "medium", "low"
    research_modules: List[str]  # Which modules should handle this
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate research opportunity data"""
        if not 0.0 <= self.priority_score <= 1.0:
            raise ValueError("Priority score must be between 0.0 and 1.0")
        
        valid_research_types = ["find_resources", "deep_dive", "monitor", "comparison"]
        if self.research_type not in valid_research_types:
            raise ValueError(f"Invalid research type: {self.research_type}")
        
        valid_urgency_levels = ["urgent", "high", "medium", "low"]
        if self.urgency not in valid_urgency_levels:
            raise ValueError(f"Invalid urgency level: {self.urgency}")
        
        valid_value_levels = ["high", "medium", "low"]
        if self.expected_value not in valid_value_levels:
            raise ValueError(f"Invalid expected value: {self.expected_value}")