"""Conversation data models"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass
class ConversationMessage:
    """Standardized conversation message format"""
    content: str
    timestamp: datetime
    role: str  # "user" or "assistant"
    conversation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate message data"""
        if self.role not in ["user", "assistant"]:
            raise ValueError(f"Invalid role: {self.role}. Must be 'user' or 'assistant'")
        
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
        
        if not self.conversation_id:
            raise ValueError("Conversation ID is required")