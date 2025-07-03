from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
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
class ConversationMessage:
    """Standardized conversation message format"""
    content: str
    timestamp: datetime
    role: str  # "user" or "assistant"
    conversation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class ResearchOpportunity:
    """Specific research task generated from interests"""
    interest_topic: str
    research_question: str
    priority_score: float
    estimated_effort: str  # "low", "medium", "high"
    suggested_tools: List[str]
    research_type: str  # "monitor", "deep_dive", "comparison", "find_resources"
    reasoning: str


class SemanticInterestAnalyzer:
    """
    Core component for analyzing conversation history and identifying 
    user interests, patterns, and research opportunities.
    """
    
    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        self.llm_client = llm_client
        self.config = config or self._default_config()
        
        # Cache for efficient incremental analysis
        self.analyzed_conversations: set = set()
        self.interest_history: Dict[str, DetectedInterest] = {}
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "analysis_window_days": 30,
            "min_mentions_for_interest": 2,
            "confidence_threshold": 0.6,
            "max_interests_per_analysis": 10,
            "research_potential_threshold": 0.5
        }
    
    def analyze_conversations(self, 
                            conversations: List[ConversationMessage],
                            incremental: bool = True) -> List[DetectedInterest]:
        """
        Main entry point: analyze conversation history for interests and patterns
        """
        
        # Filter to analysis window
        cutoff_date = datetime.now() - timedelta(days=self.config["analysis_window_days"])
        recent_conversations = [
            conv for conv in conversations 
            if conv.timestamp >= cutoff_date
        ]
        
        # Skip already analyzed conversations if incremental
        if incremental:
            new_conversations = [
                conv for conv in recent_conversations
                if conv.conversation_id not in self.analyzed_conversations
            ]
        else:
            new_conversations = recent_conversations
            self.interest_history.clear()
            self.analyzed_conversations.clear()
        
        if not new_conversations:
            return list(self.interest_history.values())
        
        # Extract and analyze interests
        extracted_interests = self._extract_semantic_interests(new_conversations)
        
        # Merge with existing interests
        self._merge_interests(extracted_interests)
        
        # Update analyzed conversation cache
        for conv in new_conversations:
            self.analyzed_conversations.add(conv.conversation_id)
        
        # Return top interests by confidence and engagement
        return self._rank_interests()
    
    def _extract_semantic_interests(self, 
                                  conversations: List[ConversationMessage]) -> List[DetectedInterest]:
        """
        Use LLM to semantically analyze conversations for interests and patterns
        """
        
        # Prepare conversation context for LLM
        conversation_text = self._format_conversations_for_analysis(conversations)
        
        # LLM analysis prompt
        analysis_prompt = self._build_analysis_prompt(conversation_text)
        
        # Call LLM (placeholder - integrate with actual LLM client)
        llm_response = self._call_llm_for_analysis(analysis_prompt)
        
        # Parse LLM response into structured interests
        return self._parse_llm_response(llm_response, conversations)
    
    def _format_conversations_for_analysis(self, 
                                         conversations: List[ConversationMessage]) -> str:
        """Format conversations for LLM analysis"""
        
        # Group by conversation and format
        conversation_groups = {}
        for msg in conversations:
            if msg.conversation_id not in conversation_groups:
                conversation_groups[msg.conversation_id] = []
            conversation_groups[msg.conversation_id].append(msg)
        
        formatted_text = ""
        for conv_id, messages in conversation_groups.items():
            formatted_text += f"\n=== Conversation {conv_id} ===\n"
            for msg in sorted(messages, key=lambda x: x.timestamp):
                formatted_text += f"{msg.role}: {msg.content}\n"
        
        return formatted_text
    
    def _build_analysis_prompt(self, conversation_text: str) -> str:
        """Build the LLM prompt for semantic interest analysis"""
        
        return f"""
Analyze the following conversation history to identify the user's interests, concerns, problems, and goals. Look for:

1. **Recurring topics** mentioned across multiple conversations
2. **Emerging interests** that are gaining attention
3. **Persistent problems** the user keeps mentioning
4. **Learning goals** or skill development interests
5. **Health concerns** or wellness topics
6. **Preferences** for products, activities, or approaches
7. **Future goals** or aspirations mentioned

For each identified interest, provide:
- **Topic**: Clear, specific topic name
- **Type**: hobby, problem, learning, health, goal, preference, or concern
- **Engagement Level**: 1-4 scale (1=casual mention, 4=persistent focus)
- **Key Phrases**: Specific phrases the user used
- **Context**: Brief summary of how this topic appeared
- **Research Potential**: 0.0-1.0 scale of how actionable/researchable this is

Return your analysis as a JSON array of interests. Be thorough but focus on topics that could benefit from autonomous research or proactive assistance.

Conversation History:
{conversation_text}

JSON Response:
"""
    
    def _call_llm_for_analysis(self, prompt: str) -> str:
        """
        Call LLM for semantic analysis
        (Placeholder - integrate with actual LLM client)
        """
        if self.llm_client:
            # Use provided LLM client
            return self.llm_client.generate(prompt)
        else:
            # Mock response for testing
            return self._mock_llm_response()
    
    def _mock_llm_response(self) -> str:
        """Mock LLM response for testing"""
        return """[
    {
        "topic": "quilting techniques",
        "type": "hobby", 
        "engagement_level": 3,
        "key_phrases": ["quilt patterns", "fabric selection", "modern quilting"],
        "context": "User has been asking about different quilting techniques and showing interest in modern patterns",
        "research_potential": 0.8
    },
    {
        "topic": "back pain management",
        "type": "health",
        "engagement_level": 2, 
        "key_phrases": ["back bothering me", "sitting too long", "ergonomic"],
        "context": "User mentioned back pain several times in relation to work setup",
        "research_potential": 0.9
    }
]"""
    
    def _parse_llm_response(self, 
                           llm_response: str, 
                           conversations: List[ConversationMessage]) -> List[DetectedInterest]:
        """Parse LLM JSON response into DetectedInterest objects"""
        
        try:
            parsed_interests = json.loads(llm_response.strip())
        except json.JSONDecodeError:
            # Attempt to extract JSON from response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                parsed_interests = json.loads(json_match.group())
            else:
                return []
        
        detected_interests = []
        
        for interest_data in parsed_interests:
            # Find first and last mentions
            first_mention, last_mention, mention_count = self._find_mention_timestamps(
                interest_data["topic"], 
                interest_data["key_phrases"], 
                conversations
            )
            
            if mention_count < self.config["min_mentions_for_interest"]:
                continue
                
            detected_interest = DetectedInterest(
                topic=interest_data["topic"],
                interest_type=InterestType(interest_data["type"]),
                engagement_level=EngagementLevel(interest_data["engagement_level"]),
                confidence_score=min(0.9, mention_count * 0.2 + interest_data["engagement_level"] * 0.15),
                first_mentioned=first_mention,
                last_mentioned=last_mention,
                mention_count=mention_count,
                key_phrases=interest_data["key_phrases"],
                context_summary=interest_data["context"],
                research_potential=interest_data["research_potential"]
            )
            
            detected_interests.append(detected_interest)
        
        return detected_interests
    
    def _find_mention_timestamps(self, 
                               topic: str, 
                               key_phrases: List[str], 
                               conversations: List[ConversationMessage]) -> Tuple[datetime, datetime, int]:
        """Find when topic was first/last mentioned and count mentions"""
        
        mentions = []
        for conv in conversations:
            # Simple keyword matching (could be enhanced with semantic similarity)
            content_lower = conv.content.lower()
            topic_lower = topic.lower()
            
            if topic_lower in content_lower or any(phrase.lower() in content_lower for phrase in key_phrases):
                mentions.append(conv.timestamp)
        
        if not mentions:
            return datetime.now(), datetime.now(), 0
        
        return min(mentions), max(mentions), len(mentions)
    
    def _merge_interests(self, new_interests: List[DetectedInterest]):
        """Merge new interests with existing interest history"""
        
        for new_interest in new_interests:
            topic_key = new_interest.topic.lower()
            
            if topic_key in self.interest_history:
                # Update existing interest
                existing = self.interest_history[topic_key]
                existing.last_mentioned = max(existing.last_mentioned, new_interest.last_mentioned)
                existing.mention_count += new_interest.mention_count
                existing.key_phrases = list(set(existing.key_phrases + new_interest.key_phrases))
                
                # Update engagement level if higher
                if new_interest.engagement_level.value > existing.engagement_level.value:
                    existing.engagement_level = new_interest.engagement_level
                
                # Recalculate confidence score
                existing.confidence_score = min(0.95, 
                    existing.mention_count * 0.15 + existing.engagement_level.value * 0.2)
            else:
                # Add new interest
                self.interest_history[topic_key] = new_interest
    
    def _rank_interests(self) -> List[DetectedInterest]:
        """Return interests ranked by relevance and research potential"""
        
        interests = list(self.interest_history.values())
        
        # Filter by confidence threshold
        filtered_interests = [
            interest for interest in interests 
            if interest.confidence_score >= self.config["confidence_threshold"]
        ]
        
        # Sort by combined score of confidence, engagement, and research potential
        def scoring_function(interest):
            return (
                interest.confidence_score * 0.4 +
                interest.engagement_level.value * 0.3 +
                interest.research_potential * 0.3
            )
        
        ranked_interests = sorted(filtered_interests, key=scoring_function, reverse=True)
        
        return ranked_interests[:self.config["max_interests_per_analysis"]]
    
    def generate_research_opportunities(self, 
                                      interests: List[DetectedInterest]) -> List[ResearchOpportunity]:
        """Generate specific research opportunities from detected interests"""
        
        opportunities = []
        
        for interest in interests:
            if interest.research_potential < self.config["research_potential_threshold"]:
                continue
            
            # Generate research opportunity based on interest type
            opportunity = self._create_research_opportunity(interest)
            if opportunity:
                opportunities.append(opportunity)
        
        return opportunities
    
    def _create_research_opportunity(self, interest: DetectedInterest) -> Optional[ResearchOpportunity]:
        """Create a research opportunity for a specific interest"""
        
        research_templates = {
            InterestType.HOBBY: {
                "question": f"Find new resources, tutorials, or communities for {interest.topic}",
                "tools": ["web_search", "youtube_search", "reddit_search"],
                "type": "find_resources"
            },
            InterestType.PROBLEM: {
                "question": f"Research solutions and expert advice for {interest.topic}",
                "tools": ["web_search", "expert_search", "product_search"],
                "type": "deep_dive"
            },
            InterestType.HEALTH: {
                "question": f"Find latest research and expert recommendations for {interest.topic}",
                "tools": ["medical_search", "web_search", "local_provider_search"],
                "type": "monitor"
            },
            InterestType.LEARNING: {
                "question": f"Curate learning path and resources for {interest.topic}",
                "tools": ["course_search", "tutorial_search", "book_search"],
                "type": "find_resources"
            }
        }
        
        template = research_templates.get(interest.interest_type)
        if not template:
            return None
        
        return ResearchOpportunity(
            interest_topic=interest.topic,
            research_question=template["question"],
            priority_score=interest.confidence_score * interest.research_potential,
            estimated_effort="medium",  # Could be more sophisticated
            suggested_tools=template["tools"],
            research_type=template["type"],
            reasoning=f"User has shown {interest.engagement_level.name.lower()} in {interest.topic} with {interest.mention_count} mentions"
        )


# Example usage and testing
if __name__ == "__main__":
    # Example conversation data
    sample_conversations = [
        ConversationMessage(
            content="I've been thinking about trying quilting. The patterns look so intricate!",
            timestamp=datetime.now() - timedelta(days=5),
            role="user",
            conversation_id="conv_1"
        ),
        ConversationMessage(
            content="My back has been bothering me from sitting at this desk all day.",
            timestamp=datetime.now() - timedelta(days=3), 
            role="user",
            conversation_id="conv_2"
        ),
        ConversationMessage(
            content="Do you know any good quilting tutorials for beginners?",
            timestamp=datetime.now() - timedelta(days=1),
            role="user", 
            conversation_id="conv_3"
        )
    ]
    
    # Initialize analyzer
    analyzer = SemanticInterestAnalyzer()
    
    # Analyze conversations
    interests = analyzer.analyze_conversations(sample_conversations)
    
    # Generate research opportunities
    opportunities = analyzer.generate_research_opportunities(interests)
    
    print("Detected Interests:")
    for interest in interests:
        print(f"- {interest.topic} ({interest.interest_type.value}) - Confidence: {interest.confidence_score:.2f}")
    
    print("\nResearch Opportunities:")
    for opp in opportunities:
        print(f"- {opp.research_question} (Priority: {opp.priority_score:.2f})")
