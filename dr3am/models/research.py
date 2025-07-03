"""Research result and insight models"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class ResearchResult:
    """Container for research findings"""
    research_id: str
    topic: str
    research_type: str
    findings: List[Dict[str, Any]]
    summary: str
    insights: List[str]
    sources: List[str]
    confidence_score: float
    research_timestamp: datetime
    expiry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate research result data"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        if not self.research_id:
            raise ValueError("Research ID is required")
        
        if not self.topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if not self.summary.strip():
            raise ValueError("Summary cannot be empty")


@dataclass
class ActionableInsight:
    """Specific action user can take based on research"""
    action_type: str  # "purchase", "read", "visit", "learn", "monitor"
    title: str
    description: str
    url: Optional[str] = None
    urgency: str = "medium"  # "low", "medium", "high", "urgent"
    estimated_value: Optional[str] = None
    deadline: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate actionable insight data"""
        valid_action_types = ["purchase", "read", "visit", "learn", "monitor", "track", "compare"]
        if self.action_type not in valid_action_types:
            raise ValueError(f"Invalid action type: {self.action_type}")
        
        valid_urgency_levels = ["low", "medium", "high", "urgent"]
        if self.urgency not in valid_urgency_levels:
            raise ValueError(f"Invalid urgency level: {self.urgency}")
        
        if not self.title.strip():
            raise ValueError("Title cannot be empty")
        
        if not self.description.strip():
            raise ValueError("Description cannot be empty")