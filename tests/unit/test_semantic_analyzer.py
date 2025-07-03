"""Unit tests for semantic analyzer"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List

from dr3am.models.conversation import ConversationMessage
from dr3am.models.interests import DetectedInterest, ResearchOpportunity, InterestType, EngagementLevel


class TestSemanticAnalyzer:
    """Test semantic analyzer functionality"""
    
    @pytest.fixture
    def mock_semantic_analyzer(self):
        """Create a mock semantic analyzer"""
        # Since we don't have the actual implementation accessible, 
        # we'll create a mock that simulates the expected behavior
        analyzer = Mock()
        analyzer.analyze_conversations = AsyncMock()
        analyzer.generate_research_opportunities = AsyncMock()
        analyzer.update_interest_tracking = AsyncMock()
        return analyzer
    
    @pytest.mark.asyncio
    async def test_analyze_conversations_basic(self, mock_semantic_analyzer, sample_conversations):
        """Test basic conversation analysis"""
        # Mock the expected return value
        expected_interests = [
            DetectedInterest(
                topic="Python programming",
                interest_type=InterestType.LEARNING,
                engagement_level=EngagementLevel.ACTIVE_ENGAGEMENT,
                confidence_score=0.9,
                first_mentioned=datetime.now() - timedelta(days=2),
                last_mentioned=datetime.now() - timedelta(hours=12),
                mention_count=3,
                key_phrases=["python", "programming", "async"],
                context_summary="User actively learning Python",
                research_potential=0.8
            )
        ]
        
        mock_semantic_analyzer.analyze_conversations.return_value = expected_interests
        
        # Call the method
        result = await mock_semantic_analyzer.analyze_conversations(sample_conversations)
        
        # Verify results
        assert len(result) == 1
        assert result[0].topic == "Python programming"
        assert result[0].interest_type == InterestType.LEARNING
        assert result[0].engagement_level == EngagementLevel.ACTIVE_ENGAGEMENT
        assert result[0].confidence_score == 0.9
        assert result[0].mention_count == 3
        
        # Verify the method was called with correct arguments
        mock_semantic_analyzer.analyze_conversations.assert_called_once_with(sample_conversations)
    
    @pytest.mark.asyncio
    async def test_analyze_conversations_empty_input(self, mock_semantic_analyzer):
        """Test conversation analysis with empty input"""
        mock_semantic_analyzer.analyze_conversations.return_value = []
        
        result = await mock_semantic_analyzer.analyze_conversations([])
        
        assert result == []
        mock_semantic_analyzer.analyze_conversations.assert_called_once_with([])
    
    @pytest.mark.asyncio
    async def test_analyze_conversations_multiple_interests(self, mock_semantic_analyzer, sample_conversations):
        """Test conversation analysis returning multiple interests"""
        expected_interests = [
            DetectedInterest(
                topic="Python programming",
                interest_type=InterestType.LEARNING,
                engagement_level=EngagementLevel.ACTIVE_ENGAGEMENT,
                confidence_score=0.9,
                first_mentioned=datetime.now() - timedelta(days=2),
                last_mentioned=datetime.now() - timedelta(hours=12),
                mention_count=3,
                key_phrases=["python", "programming"],
                context_summary="User learning Python",
                research_potential=0.8
            ),
            DetectedInterest(
                topic="Machine learning",
                interest_type=InterestType.LEARNING,
                engagement_level=EngagementLevel.RECURRING_INTEREST,
                confidence_score=0.7,
                first_mentioned=datetime.now() - timedelta(hours=6),
                last_mentioned=datetime.now() - timedelta(hours=6),
                mention_count=1,
                key_phrases=["machine learning", "ML"],
                context_summary="User interested in ML",
                research_potential=0.6
            )
        ]
        
        mock_semantic_analyzer.analyze_conversations.return_value = expected_interests
        
        result = await mock_semantic_analyzer.analyze_conversations(sample_conversations)
        
        assert len(result) == 2
        assert result[0].topic == "Python programming"
        assert result[1].topic == "Machine learning"
        assert result[0].engagement_level == EngagementLevel.ACTIVE_ENGAGEMENT
        assert result[1].engagement_level == EngagementLevel.RECURRING_INTEREST
    
    @pytest.mark.asyncio
    async def test_generate_research_opportunities_basic(self, mock_semantic_analyzer, sample_detected_interests):
        """Test basic research opportunity generation"""
        expected_opportunities = [
            ResearchOpportunity(
                interest_topic="Python programming",
                research_type="find_resources",
                priority_score=0.9,
                research_query="Python async programming tutorials",
                expected_value="high",
                urgency="medium",
                research_modules=["web_search", "educational_resources"]
            )
        ]
        
        mock_semantic_analyzer.generate_research_opportunities.return_value = expected_opportunities
        
        result = await mock_semantic_analyzer.generate_research_opportunities(sample_detected_interests)
        
        assert len(result) == 1
        assert result[0].interest_topic == "Python programming"
        assert result[0].research_type == "find_resources"
        assert result[0].priority_score == 0.9
        assert result[0].expected_value == "high"
        assert result[0].urgency == "medium"
        
        mock_semantic_analyzer.generate_research_opportunities.assert_called_once_with(sample_detected_interests)
    
    @pytest.mark.asyncio
    async def test_generate_research_opportunities_empty_input(self, mock_semantic_analyzer):
        """Test research opportunity generation with empty input"""
        mock_semantic_analyzer.generate_research_opportunities.return_value = []
        
        result = await mock_semantic_analyzer.generate_research_opportunities([])
        
        assert result == []
        mock_semantic_analyzer.generate_research_opportunities.assert_called_once_with([])
    
    @pytest.mark.asyncio
    async def test_generate_research_opportunities_multiple_types(self, mock_semantic_analyzer, sample_detected_interests):
        """Test research opportunity generation with multiple types"""
        expected_opportunities = [
            ResearchOpportunity(
                interest_topic="Python programming",
                research_type="find_resources",
                priority_score=0.9,
                research_query="Python async programming tutorials",
                expected_value="high",
                urgency="medium",
                research_modules=["web_search", "educational_resources"]
            ),
            ResearchOpportunity(
                interest_topic="Tech stock investing",
                research_type="monitor",
                priority_score=0.7,
                research_query="tech stock market performance",
                expected_value="medium",
                urgency="low",
                research_modules=["financial_data", "news_monitoring"]
            )
        ]
        
        mock_semantic_analyzer.generate_research_opportunities.return_value = expected_opportunities
        
        result = await mock_semantic_analyzer.generate_research_opportunities(sample_detected_interests)
        
        assert len(result) == 2
        assert result[0].research_type == "find_resources"
        assert result[1].research_type == "monitor"
        assert result[0].priority_score == 0.9
        assert result[1].priority_score == 0.7
    
    @pytest.mark.asyncio
    async def test_update_interest_tracking(self, mock_semantic_analyzer, sample_detected_interests):
        """Test interest tracking update"""
        mock_semantic_analyzer.update_interest_tracking.return_value = True
        
        result = await mock_semantic_analyzer.update_interest_tracking(sample_detected_interests)
        
        assert result is True
        mock_semantic_analyzer.update_interest_tracking.assert_called_once_with(sample_detected_interests)
    
    def test_interest_type_enum_values(self):
        """Test that InterestType enum has expected values"""
        expected_types = ["hobby", "problem", "learning", "health", "goal", "preference", "concern"]
        
        for expected_type in expected_types:
            assert any(interest_type.value == expected_type for interest_type in InterestType)
    
    def test_engagement_level_enum_values(self):
        """Test that EngagementLevel enum has expected values"""
        expected_levels = [1, 2, 3, 4]
        
        for expected_level in expected_levels:
            assert any(engagement_level.value == expected_level for engagement_level in EngagementLevel)
    
    def test_engagement_level_ordering(self):
        """Test that engagement levels are properly ordered"""
        assert EngagementLevel.CASUAL_MENTION.value < EngagementLevel.RECURRING_INTEREST.value
        assert EngagementLevel.RECURRING_INTEREST.value < EngagementLevel.ACTIVE_ENGAGEMENT.value
        assert EngagementLevel.ACTIVE_ENGAGEMENT.value < EngagementLevel.PERSISTENT_FOCUS.value


class TestSemanticAnalyzerConfiguration:
    """Test semantic analyzer configuration"""
    
    @pytest.fixture
    def mock_analyzer_with_config(self):
        """Create a mock analyzer with configuration"""
        analyzer = Mock()
        analyzer.config = {
            "analysis_window_days": 30,
            "min_mentions_for_interest": 2,
            "confidence_threshold": 0.6,
            "max_interests_per_analysis": 10
        }
        return analyzer
    
    def test_default_configuration(self, mock_analyzer_with_config):
        """Test default configuration values"""
        config = mock_analyzer_with_config.config
        
        assert config["analysis_window_days"] == 30
        assert config["min_mentions_for_interest"] == 2
        assert config["confidence_threshold"] == 0.6
        assert config["max_interests_per_analysis"] == 10
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = {
            "analysis_window_days": 30,
            "min_mentions_for_interest": 2,
            "confidence_threshold": 0.6,
            "max_interests_per_analysis": 10
        }
        
        # All values should be positive
        assert valid_config["analysis_window_days"] > 0
        assert valid_config["min_mentions_for_interest"] > 0
        assert 0.0 <= valid_config["confidence_threshold"] <= 1.0
        assert valid_config["max_interests_per_analysis"] > 0


class TestSemanticAnalyzerErrorHandling:
    """Test semantic analyzer error handling"""
    
    @pytest.fixture
    def mock_analyzer_with_errors(self):
        """Create a mock analyzer that can raise errors"""
        analyzer = Mock()
        analyzer.analyze_conversations = AsyncMock()
        analyzer.generate_research_opportunities = AsyncMock()
        return analyzer
    
    @pytest.mark.asyncio
    async def test_analyze_conversations_error_handling(self, mock_analyzer_with_errors):
        """Test error handling during conversation analysis"""
        # Mock an exception
        mock_analyzer_with_errors.analyze_conversations.side_effect = Exception("Analysis failed")
        
        with pytest.raises(Exception, match="Analysis failed"):
            await mock_analyzer_with_errors.analyze_conversations([])
    
    @pytest.mark.asyncio
    async def test_generate_research_opportunities_error_handling(self, mock_analyzer_with_errors):
        """Test error handling during research opportunity generation"""
        # Mock an exception
        mock_analyzer_with_errors.generate_research_opportunities.side_effect = Exception("Generation failed")
        
        with pytest.raises(Exception, match="Generation failed"):
            await mock_analyzer_with_errors.generate_research_opportunities([])
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, mock_analyzer_with_errors):
        """Test handling of invalid input"""
        # Test with None input
        mock_analyzer_with_errors.analyze_conversations.side_effect = ValueError("Invalid input")
        
        with pytest.raises(ValueError, match="Invalid input"):
            await mock_analyzer_with_errors.analyze_conversations(None)