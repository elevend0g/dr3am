"""Test configuration and fixtures for dr3am tests"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

from dr3am.models.conversation import ConversationMessage
from dr3am.models.interests import DetectedInterest, ResearchOpportunity
from dr3am.models.research import ResearchResult, ActionableInsight
from dr3am.core.semantic_analyzer import InterestType, EngagementLevel


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_conversations() -> List[ConversationMessage]:
    """Sample conversation data for testing"""
    return [
        ConversationMessage(
            content="I'm really interested in learning Python programming",
            timestamp=datetime.now() - timedelta(days=2),
            role="user",
            conversation_id="conv_1",
            metadata={"session_id": "session_1"}
        ),
        ConversationMessage(
            content="I've been struggling with async programming concepts",
            timestamp=datetime.now() - timedelta(days=1),
            role="user", 
            conversation_id="conv_1",
            metadata={"session_id": "session_1"}
        ),
        ConversationMessage(
            content="Do you have any recommendations for Python tutorials?",
            timestamp=datetime.now() - timedelta(hours=12),
            role="user",
            conversation_id="conv_1", 
            metadata={"session_id": "session_1"}
        ),
        ConversationMessage(
            content="I'm also interested in machine learning applications",
            timestamp=datetime.now() - timedelta(hours=6),
            role="user",
            conversation_id="conv_2",
            metadata={"session_id": "session_2"}
        ),
        ConversationMessage(
            content="Been looking into investing in tech stocks lately",
            timestamp=datetime.now() - timedelta(hours=3),
            role="user",
            conversation_id="conv_2",
            metadata={"session_id": "session_2"}
        ),
    ]


@pytest.fixture
def sample_detected_interests() -> List[DetectedInterest]:
    """Sample detected interests for testing"""
    return [
        DetectedInterest(
            topic="Python programming",
            interest_type=InterestType.LEARNING,
            engagement_level=EngagementLevel.ACTIVE_ENGAGEMENT,
            confidence_score=0.9,
            first_mentioned=datetime.now() - timedelta(days=2),
            last_mentioned=datetime.now() - timedelta(hours=12),
            mention_count=3,
            key_phrases=["python", "programming", "tutorials", "async"],
            context_summary="User is actively learning Python programming with focus on async concepts",
            research_potential=0.8,
            related_topics=["async programming", "python tutorials"]
        ),
        DetectedInterest(
            topic="Machine learning",
            interest_type=InterestType.LEARNING,
            engagement_level=EngagementLevel.RECURRING_INTEREST,
            confidence_score=0.7,
            first_mentioned=datetime.now() - timedelta(hours=6),
            last_mentioned=datetime.now() - timedelta(hours=6),
            mention_count=1,
            key_phrases=["machine learning", "applications"],
            context_summary="User expressed interest in machine learning applications",
            research_potential=0.6,
            related_topics=["AI", "data science"]
        ),
        DetectedInterest(
            topic="Tech stock investing",
            interest_type=InterestType.GOAL,
            engagement_level=EngagementLevel.CASUAL_MENTION,
            confidence_score=0.6,
            first_mentioned=datetime.now() - timedelta(hours=3),
            last_mentioned=datetime.now() - timedelta(hours=3),
            mention_count=1,
            key_phrases=["investing", "tech stocks"],
            context_summary="User mentioned interest in tech stock investing",
            research_potential=0.7,
            related_topics=["stock market", "technology companies"]
        ),
    ]


@pytest.fixture
def sample_research_opportunities() -> List[ResearchOpportunity]:
    """Sample research opportunities for testing"""
    return [
        ResearchOpportunity(
            interest_topic="Python programming",
            research_type="find_resources",
            priority_score=0.9,
            research_query="beginner Python async programming tutorials",
            expected_value="high",
            urgency="medium",
            research_modules=["web_search", "educational_resources"],
            context={"skill_level": "beginner", "focus_area": "async programming"}
        ),
        ResearchOpportunity(
            interest_topic="Tech stock investing",
            research_type="monitor",
            priority_score=0.7,
            research_query="tech stock performance and analysis",
            expected_value="medium",
            urgency="low",
            research_modules=["financial_data", "news_monitoring"],
            context={"investment_type": "stocks", "sector": "technology"}
        ),
    ]


@pytest.fixture
def sample_research_results() -> List[ResearchResult]:
    """Sample research results for testing"""
    return [
        ResearchResult(
            research_id="research_1",
            topic="Python programming",
            research_type="find_resources",
            findings=[
                {
                    "title": "Real Python - Async Programming",
                    "url": "https://realpython.com/async-io-python/",
                    "description": "Comprehensive guide to async programming in Python",
                    "relevance_score": 0.95
                },
                {
                    "title": "Python.org - Asyncio Documentation",
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "description": "Official Python asyncio documentation",
                    "relevance_score": 0.90
                }
            ],
            summary="Found high-quality resources for learning Python async programming",
            insights=[
                "Real Python provides practical examples and exercises",
                "Official documentation is comprehensive but may be advanced",
                "Consider starting with practical examples before diving into theory"
            ],
            sources=[
                "https://realpython.com/async-io-python/",
                "https://docs.python.org/3/library/asyncio.html"
            ],
            confidence_score=0.9,
            research_timestamp=datetime.now() - timedelta(hours=1),
            metadata={"search_terms": ["python", "async", "tutorial"], "api_calls": 2}
        ),
    ]


@pytest.fixture
def sample_actionable_insights() -> List[ActionableInsight]:
    """Sample actionable insights for testing"""
    return [
        ActionableInsight(
            action_type="learn",
            title="Start with Real Python Async Tutorial",
            description="Begin your async programming journey with Real Python's comprehensive tutorial",
            url="https://realpython.com/async-io-python/",
            urgency="medium",
            estimated_value="high learning value",
            deadline=datetime.now() + timedelta(days=7)
        ),
        ActionableInsight(
            action_type="read",
            title="Review Official Asyncio Documentation",
            description="Reference the official Python asyncio documentation for in-depth understanding",
            url="https://docs.python.org/3/library/asyncio.html",
            urgency="low",
            estimated_value="reference material",
            deadline=None
        ),
    ]


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    client = AsyncMock()
    client.generate_response = AsyncMock(return_value="Mock LLM response")
    return client


@pytest.fixture
def mock_api_client():
    """Mock API client for testing"""
    client = AsyncMock()
    client.search = AsyncMock(return_value={
        "success": True,
        "results": [
            {
                "title": "Sample Result",
                "url": "https://example.com",
                "snippet": "Sample description",
                "relevance": 0.8
            }
        ]
    })
    return client


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration"""
    return {
        "semantic_analysis": {
            "analysis_window_days": 30,
            "min_mentions_for_interest": 2,
            "confidence_threshold": 0.6,
            "max_interests_per_analysis": 10
        },
        "research_execution": {
            "auto_research_enabled": True,
            "research_cooldown_hours": 24,
            "daily_api_budget": 10.0,
            "max_concurrent_research": 3
        },
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False
        },
        "logging": {
            "level": "INFO",
            "format": "json"
        }
    }


# Test utilities
@pytest.fixture
def assert_datetime_close():
    """Utility to assert datetime objects are close to each other"""
    def _assert_close(dt1: datetime, dt2: datetime, delta_seconds: int = 5):
        assert abs((dt1 - dt2).total_seconds()) < delta_seconds
    return _assert_close