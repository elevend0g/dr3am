"""Unit tests for dr3am data models"""

import pytest
from datetime import datetime, timedelta
from dr3am.models.conversation import ConversationMessage
from dr3am.models.interests import DetectedInterest, ResearchOpportunity, InterestType, EngagementLevel
from dr3am.models.research import ResearchResult, ActionableInsight


class TestConversationMessage:
    """Test ConversationMessage model"""
    
    def test_valid_message_creation(self):
        """Test creating a valid conversation message"""
        message = ConversationMessage(
            content="Hello world",
            timestamp=datetime.now(),
            role="user",
            conversation_id="conv_1"
        )
        
        assert message.content == "Hello world"
        assert message.role == "user"
        assert message.conversation_id == "conv_1"
        assert isinstance(message.metadata, dict)
    
    def test_invalid_role_raises_error(self):
        """Test that invalid role raises ValueError"""
        with pytest.raises(ValueError, match="Invalid role"):
            ConversationMessage(
                content="Hello world",
                timestamp=datetime.now(),
                role="invalid_role",
                conversation_id="conv_1"
            )
    
    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError"""
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            ConversationMessage(
                content="   ",
                timestamp=datetime.now(),
                role="user",
                conversation_id="conv_1"
            )
    
    def test_missing_conversation_id_raises_error(self):
        """Test that missing conversation ID raises ValueError"""
        with pytest.raises(ValueError, match="Conversation ID is required"):
            ConversationMessage(
                content="Hello world",
                timestamp=datetime.now(),
                role="user",
                conversation_id=""
            )


class TestDetectedInterest:
    """Test DetectedInterest model"""
    
    def test_valid_interest_creation(self):
        """Test creating a valid detected interest"""
        now = datetime.now()
        interest = DetectedInterest(
            topic="Python programming",
            interest_type=InterestType.LEARNING,
            engagement_level=EngagementLevel.ACTIVE_ENGAGEMENT,
            confidence_score=0.9,
            first_mentioned=now - timedelta(days=1),
            last_mentioned=now,
            mention_count=3,
            key_phrases=["python", "programming"],
            context_summary="User learning Python",
            research_potential=0.8
        )
        
        assert interest.topic == "Python programming"
        assert interest.interest_type == InterestType.LEARNING
        assert interest.engagement_level == EngagementLevel.ACTIVE_ENGAGEMENT
        assert interest.confidence_score == 0.9
        assert interest.mention_count == 3
        assert interest.research_potential == 0.8
    
    def test_invalid_confidence_score_raises_error(self):
        """Test that invalid confidence score raises ValueError"""
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            DetectedInterest(
                topic="Python programming",
                interest_type=InterestType.LEARNING,
                engagement_level=EngagementLevel.ACTIVE_ENGAGEMENT,
                confidence_score=1.5,  # Invalid
                first_mentioned=datetime.now(),
                last_mentioned=datetime.now(),
                mention_count=1,
                key_phrases=["python"],
                context_summary="Test",
                research_potential=0.8
            )
    
    def test_invalid_research_potential_raises_error(self):
        """Test that invalid research potential raises ValueError"""
        with pytest.raises(ValueError, match="Research potential must be between 0.0 and 1.0"):
            DetectedInterest(
                topic="Python programming",
                interest_type=InterestType.LEARNING,
                engagement_level=EngagementLevel.ACTIVE_ENGAGEMENT,
                confidence_score=0.9,
                first_mentioned=datetime.now(),
                last_mentioned=datetime.now(),
                mention_count=1,
                key_phrases=["python"],
                context_summary="Test",
                research_potential=1.5  # Invalid
            )
    
    def test_invalid_mention_count_raises_error(self):
        """Test that invalid mention count raises ValueError"""
        with pytest.raises(ValueError, match="Mention count must be at least 1"):
            DetectedInterest(
                topic="Python programming",
                interest_type=InterestType.LEARNING,
                engagement_level=EngagementLevel.ACTIVE_ENGAGEMENT,
                confidence_score=0.9,
                first_mentioned=datetime.now(),
                last_mentioned=datetime.now(),
                mention_count=0,  # Invalid
                key_phrases=["python"],
                context_summary="Test",
                research_potential=0.8
            )
    
    def test_invalid_date_order_raises_error(self):
        """Test that invalid date order raises ValueError"""
        now = datetime.now()
        with pytest.raises(ValueError, match="First mentioned date cannot be after last mentioned date"):
            DetectedInterest(
                topic="Python programming",
                interest_type=InterestType.LEARNING,
                engagement_level=EngagementLevel.ACTIVE_ENGAGEMENT,
                confidence_score=0.9,
                first_mentioned=now,
                last_mentioned=now - timedelta(days=1),  # Invalid order
                mention_count=1,
                key_phrases=["python"],
                context_summary="Test",
                research_potential=0.8
            )


class TestResearchOpportunity:
    """Test ResearchOpportunity model"""
    
    def test_valid_opportunity_creation(self):
        """Test creating a valid research opportunity"""
        opportunity = ResearchOpportunity(
            interest_topic="Python programming",
            research_type="find_resources",
            priority_score=0.9,
            research_query="Python tutorial resources",
            expected_value="high",
            urgency="medium",
            research_modules=["web_search", "educational_resources"]
        )
        
        assert opportunity.interest_topic == "Python programming"
        assert opportunity.research_type == "find_resources"
        assert opportunity.priority_score == 0.9
        assert opportunity.expected_value == "high"
        assert opportunity.urgency == "medium"
        assert len(opportunity.research_modules) == 2
    
    def test_invalid_priority_score_raises_error(self):
        """Test that invalid priority score raises ValueError"""
        with pytest.raises(ValueError, match="Priority score must be between 0.0 and 1.0"):
            ResearchOpportunity(
                interest_topic="Python programming",
                research_type="find_resources",
                priority_score=1.5,  # Invalid
                research_query="Python tutorial resources",
                expected_value="high",
                urgency="medium",
                research_modules=["web_search"]
            )
    
    def test_invalid_research_type_raises_error(self):
        """Test that invalid research type raises ValueError"""
        with pytest.raises(ValueError, match="Invalid research type"):
            ResearchOpportunity(
                interest_topic="Python programming",
                research_type="invalid_type",  # Invalid
                priority_score=0.9,
                research_query="Python tutorial resources",
                expected_value="high",
                urgency="medium",
                research_modules=["web_search"]
            )
    
    def test_invalid_urgency_raises_error(self):
        """Test that invalid urgency raises ValueError"""
        with pytest.raises(ValueError, match="Invalid urgency level"):
            ResearchOpportunity(
                interest_topic="Python programming",
                research_type="find_resources",
                priority_score=0.9,
                research_query="Python tutorial resources",
                expected_value="high",
                urgency="invalid_urgency",  # Invalid
                research_modules=["web_search"]
            )
    
    def test_invalid_expected_value_raises_error(self):
        """Test that invalid expected value raises ValueError"""
        with pytest.raises(ValueError, match="Invalid expected value"):
            ResearchOpportunity(
                interest_topic="Python programming",
                research_type="find_resources",
                priority_score=0.9,
                research_query="Python tutorial resources",
                expected_value="invalid_value",  # Invalid
                urgency="medium",
                research_modules=["web_search"]
            )


class TestResearchResult:
    """Test ResearchResult model"""
    
    def test_valid_result_creation(self):
        """Test creating a valid research result"""
        result = ResearchResult(
            research_id="research_1",
            topic="Python programming",
            research_type="find_resources",
            findings=[{"title": "Python Tutorial", "url": "https://example.com"}],
            summary="Found Python tutorials",
            insights=["Python is popular", "Many resources available"],
            sources=["https://example.com"],
            confidence_score=0.9,
            research_timestamp=datetime.now()
        )
        
        assert result.research_id == "research_1"
        assert result.topic == "Python programming"
        assert result.research_type == "find_resources"
        assert len(result.findings) == 1
        assert len(result.insights) == 2
        assert result.confidence_score == 0.9
    
    def test_invalid_confidence_score_raises_error(self):
        """Test that invalid confidence score raises ValueError"""
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            ResearchResult(
                research_id="research_1",
                topic="Python programming",
                research_type="find_resources",
                findings=[],
                summary="Test summary",
                insights=[],
                sources=[],
                confidence_score=1.5,  # Invalid
                research_timestamp=datetime.now()
            )
    
    def test_empty_research_id_raises_error(self):
        """Test that empty research ID raises ValueError"""
        with pytest.raises(ValueError, match="Research ID is required"):
            ResearchResult(
                research_id="",  # Invalid
                topic="Python programming",
                research_type="find_resources",
                findings=[],
                summary="Test summary",
                insights=[],
                sources=[],
                confidence_score=0.9,
                research_timestamp=datetime.now()
            )
    
    def test_empty_topic_raises_error(self):
        """Test that empty topic raises ValueError"""
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            ResearchResult(
                research_id="research_1",
                topic="   ",  # Invalid
                research_type="find_resources",
                findings=[],
                summary="Test summary",
                insights=[],
                sources=[],
                confidence_score=0.9,
                research_timestamp=datetime.now()
            )
    
    def test_empty_summary_raises_error(self):
        """Test that empty summary raises ValueError"""
        with pytest.raises(ValueError, match="Summary cannot be empty"):
            ResearchResult(
                research_id="research_1",
                topic="Python programming",
                research_type="find_resources",
                findings=[],
                summary="   ",  # Invalid
                insights=[],
                sources=[],
                confidence_score=0.9,
                research_timestamp=datetime.now()
            )


class TestActionableInsight:
    """Test ActionableInsight model"""
    
    def test_valid_insight_creation(self):
        """Test creating a valid actionable insight"""
        insight = ActionableInsight(
            action_type="learn",
            title="Learn Python Basics",
            description="Start with Python fundamentals",
            url="https://example.com/python-tutorial",
            urgency="medium",
            estimated_value="high learning value"
        )
        
        assert insight.action_type == "learn"
        assert insight.title == "Learn Python Basics"
        assert insight.description == "Start with Python fundamentals"
        assert insight.url == "https://example.com/python-tutorial"
        assert insight.urgency == "medium"
        assert insight.estimated_value == "high learning value"
    
    def test_invalid_action_type_raises_error(self):
        """Test that invalid action type raises ValueError"""
        with pytest.raises(ValueError, match="Invalid action type"):
            ActionableInsight(
                action_type="invalid_action",  # Invalid
                title="Learn Python Basics",
                description="Start with Python fundamentals",
                urgency="medium"
            )
    
    def test_invalid_urgency_raises_error(self):
        """Test that invalid urgency raises ValueError"""
        with pytest.raises(ValueError, match="Invalid urgency level"):
            ActionableInsight(
                action_type="learn",
                title="Learn Python Basics",
                description="Start with Python fundamentals",
                urgency="invalid_urgency"  # Invalid
            )
    
    def test_empty_title_raises_error(self):
        """Test that empty title raises ValueError"""
        with pytest.raises(ValueError, match="Title cannot be empty"):
            ActionableInsight(
                action_type="learn",
                title="   ",  # Invalid
                description="Start with Python fundamentals",
                urgency="medium"
            )
    
    def test_empty_description_raises_error(self):
        """Test that empty description raises ValueError"""
        with pytest.raises(ValueError, match="Description cannot be empty"):
            ActionableInsight(
                action_type="learn",
                title="Learn Python Basics",
                description="   ",  # Invalid
                urgency="medium"
            )