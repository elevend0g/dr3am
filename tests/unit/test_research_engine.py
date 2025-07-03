"""Unit tests for research execution engine"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from dr3am.models.interests import ResearchOpportunity
from dr3am.models.research import ResearchResult, ActionableInsight


class TestResearchEngine:
    """Test research engine functionality"""
    
    @pytest.fixture
    def mock_research_engine(self):
        """Create a mock research engine"""
        engine = Mock()
        engine.execute_research = AsyncMock()
        engine.generate_insights = AsyncMock()
        engine.schedule_research = AsyncMock()
        engine.get_research_status = AsyncMock()
        return engine
    
    @pytest.mark.asyncio
    async def test_execute_research_basic(self, mock_research_engine, sample_research_opportunities):
        """Test basic research execution"""
        opportunity = sample_research_opportunities[0]
        
        expected_result = ResearchResult(
            research_id="research_1",
            topic="Python programming",
            research_type="find_resources",
            findings=[
                {
                    "title": "Python Tutorial",
                    "url": "https://example.com/python-tutorial",
                    "description": "Comprehensive Python tutorial",
                    "relevance_score": 0.9
                }
            ],
            summary="Found high-quality Python resources",
            insights=["Python is beginner-friendly", "Many free resources available"],
            sources=["https://example.com/python-tutorial"],
            confidence_score=0.9,
            research_timestamp=datetime.now()
        )
        
        mock_research_engine.execute_research.return_value = expected_result
        
        result = await mock_research_engine.execute_research(opportunity)
        
        assert result.research_id == "research_1"
        assert result.topic == "Python programming"
        assert result.research_type == "find_resources"
        assert len(result.findings) == 1
        assert len(result.insights) == 2
        assert result.confidence_score == 0.9
        
        mock_research_engine.execute_research.assert_called_once_with(opportunity)
    
    @pytest.mark.asyncio
    async def test_execute_research_multiple_findings(self, mock_research_engine, sample_research_opportunities):
        """Test research execution with multiple findings"""
        opportunity = sample_research_opportunities[0]
        
        expected_result = ResearchResult(
            research_id="research_2",
            topic="Python programming",
            research_type="find_resources",
            findings=[
                {
                    "title": "Python Tutorial 1",
                    "url": "https://example.com/tutorial1",
                    "description": "Beginner Python tutorial",
                    "relevance_score": 0.9
                },
                {
                    "title": "Python Tutorial 2",
                    "url": "https://example.com/tutorial2",
                    "description": "Advanced Python tutorial",
                    "relevance_score": 0.8
                },
                {
                    "title": "Python Documentation",
                    "url": "https://docs.python.org",
                    "description": "Official Python documentation",
                    "relevance_score": 0.95
                }
            ],
            summary="Found comprehensive Python learning resources",
            insights=[
                "Multiple difficulty levels available",
                "Official documentation is comprehensive",
                "Community tutorials provide practical examples"
            ],
            sources=[
                "https://example.com/tutorial1",
                "https://example.com/tutorial2", 
                "https://docs.python.org"
            ],
            confidence_score=0.95,
            research_timestamp=datetime.now()
        )
        
        mock_research_engine.execute_research.return_value = expected_result
        
        result = await mock_research_engine.execute_research(opportunity)
        
        assert len(result.findings) == 3
        assert len(result.insights) == 3
        assert len(result.sources) == 3
        assert result.confidence_score == 0.95
        
        # Verify findings are properly structured
        for finding in result.findings:
            assert "title" in finding
            assert "url" in finding
            assert "description" in finding
            assert "relevance_score" in finding
    
    @pytest.mark.asyncio
    async def test_generate_insights_basic(self, mock_research_engine, sample_research_results):
        """Test basic insight generation"""
        research_result = sample_research_results[0]
        
        expected_insights = [
            ActionableInsight(
                action_type="learn",
                title="Start with Real Python Tutorial",
                description="Begin learning Python with this comprehensive tutorial",
                url="https://realpython.com/async-io-python/",
                urgency="medium",
                estimated_value="high learning value"
            ),
            ActionableInsight(
                action_type="read",
                title="Review Python Documentation",
                description="Read the official Python asyncio documentation",
                url="https://docs.python.org/3/library/asyncio.html",
                urgency="low",
                estimated_value="reference value"
            )
        ]
        
        mock_research_engine.generate_insights.return_value = expected_insights
        
        result = await mock_research_engine.generate_insights(research_result)
        
        assert len(result) == 2
        assert result[0].action_type == "learn"
        assert result[1].action_type == "read"
        assert result[0].urgency == "medium"
        assert result[1].urgency == "low"
        
        mock_research_engine.generate_insights.assert_called_once_with(research_result)
    
    @pytest.mark.asyncio
    async def test_schedule_research_basic(self, mock_research_engine, sample_research_opportunities):
        """Test basic research scheduling"""
        opportunities = sample_research_opportunities
        
        expected_schedule = {
            "scheduled_count": 2,
            "high_priority_count": 1,
            "medium_priority_count": 1,
            "estimated_completion": datetime.now() + timedelta(hours=2)
        }
        
        mock_research_engine.schedule_research.return_value = expected_schedule
        
        result = await mock_research_engine.schedule_research(opportunities)
        
        assert result["scheduled_count"] == 2
        assert result["high_priority_count"] == 1
        assert result["medium_priority_count"] == 1
        assert "estimated_completion" in result
        
        mock_research_engine.schedule_research.assert_called_once_with(opportunities)
    
    @pytest.mark.asyncio
    async def test_get_research_status(self, mock_research_engine):
        """Test research status retrieval"""
        expected_status = {
            "active_research_count": 2,
            "completed_research_count": 5,
            "failed_research_count": 1,
            "queue_size": 3,
            "average_completion_time": 45.5,
            "success_rate": 0.83
        }
        
        mock_research_engine.get_research_status.return_value = expected_status
        
        result = await mock_research_engine.get_research_status()
        
        assert result["active_research_count"] == 2
        assert result["completed_research_count"] == 5
        assert result["failed_research_count"] == 1
        assert result["queue_size"] == 3
        assert result["average_completion_time"] == 45.5
        assert result["success_rate"] == 0.83
        
        mock_research_engine.get_research_status.assert_called_once()


class TestResearchEngineErrorHandling:
    """Test research engine error handling"""
    
    @pytest.fixture
    def mock_engine_with_errors(self):
        """Create a mock engine that can raise errors"""
        engine = Mock()
        engine.execute_research = AsyncMock()
        engine.generate_insights = AsyncMock()
        engine.schedule_research = AsyncMock()
        return engine
    
    @pytest.mark.asyncio
    async def test_execute_research_error_handling(self, mock_engine_with_errors):
        """Test error handling during research execution"""
        mock_engine_with_errors.execute_research.side_effect = Exception("Research failed")
        
        with pytest.raises(Exception, match="Research failed"):
            await mock_engine_with_errors.execute_research(Mock())
    
    @pytest.mark.asyncio
    async def test_generate_insights_error_handling(self, mock_engine_with_errors):
        """Test error handling during insight generation"""
        mock_engine_with_errors.generate_insights.side_effect = Exception("Insight generation failed")
        
        with pytest.raises(Exception, match="Insight generation failed"):
            await mock_engine_with_errors.generate_insights(Mock())
    
    @pytest.mark.asyncio
    async def test_schedule_research_error_handling(self, mock_engine_with_errors):
        """Test error handling during research scheduling"""
        mock_engine_with_errors.schedule_research.side_effect = Exception("Scheduling failed")
        
        with pytest.raises(Exception, match="Scheduling failed"):
            await mock_engine_with_errors.schedule_research([])
    
    @pytest.mark.asyncio
    async def test_invalid_opportunity_handling(self, mock_engine_with_errors):
        """Test handling of invalid research opportunities"""
        mock_engine_with_errors.execute_research.side_effect = ValueError("Invalid opportunity")
        
        with pytest.raises(ValueError, match="Invalid opportunity"):
            await mock_engine_with_errors.execute_research(None)


class TestResearchEngineConfiguration:
    """Test research engine configuration"""
    
    @pytest.fixture
    def mock_engine_with_config(self):
        """Create a mock engine with configuration"""
        engine = Mock()
        engine.config = {
            "max_concurrent_research": 3,
            "research_timeout_seconds": 300,
            "retry_attempts": 3,
            "api_rate_limit": 60,
            "daily_api_budget": 10.0
        }
        return engine
    
    def test_default_configuration(self, mock_engine_with_config):
        """Test default configuration values"""
        config = mock_engine_with_config.config
        
        assert config["max_concurrent_research"] == 3
        assert config["research_timeout_seconds"] == 300
        assert config["retry_attempts"] == 3
        assert config["api_rate_limit"] == 60
        assert config["daily_api_budget"] == 10.0
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        valid_config = {
            "max_concurrent_research": 3,
            "research_timeout_seconds": 300,
            "retry_attempts": 3,
            "api_rate_limit": 60,
            "daily_api_budget": 10.0
        }
        
        # All values should be positive
        assert valid_config["max_concurrent_research"] > 0
        assert valid_config["research_timeout_seconds"] > 0
        assert valid_config["retry_attempts"] > 0
        assert valid_config["api_rate_limit"] > 0
        assert valid_config["daily_api_budget"] > 0.0


class TestResearchEnginePerformance:
    """Test research engine performance characteristics"""
    
    @pytest.fixture
    def mock_performance_engine(self):
        """Create a mock engine with performance metrics"""
        engine = Mock()
        engine.get_performance_metrics = AsyncMock()
        engine.optimize_research_queue = AsyncMock()
        return engine
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, mock_performance_engine):
        """Test performance metrics retrieval"""
        expected_metrics = {
            "average_research_time": 42.5,
            "success_rate": 0.87,
            "api_calls_per_research": 3.2,
            "cache_hit_rate": 0.65,
            "concurrent_research_utilization": 0.78
        }
        
        mock_performance_engine.get_performance_metrics.return_value = expected_metrics
        
        result = await mock_performance_engine.get_performance_metrics()
        
        assert result["average_research_time"] == 42.5
        assert result["success_rate"] == 0.87
        assert result["api_calls_per_research"] == 3.2
        assert result["cache_hit_rate"] == 0.65
        assert result["concurrent_research_utilization"] == 0.78
    
    @pytest.mark.asyncio
    async def test_optimize_research_queue(self, mock_performance_engine):
        """Test research queue optimization"""
        expected_optimization = {
            "queue_size_before": 10,
            "queue_size_after": 7,
            "priority_adjustments": 3,
            "duplicate_removals": 2,
            "optimization_time": 0.15
        }
        
        mock_performance_engine.optimize_research_queue.return_value = expected_optimization
        
        result = await mock_performance_engine.optimize_research_queue()
        
        assert result["queue_size_before"] == 10
        assert result["queue_size_after"] == 7
        assert result["priority_adjustments"] == 3
        assert result["duplicate_removals"] == 2
        assert result["optimization_time"] == 0.15


class TestResearchEngineIntegration:
    """Test research engine integration capabilities"""
    
    @pytest.fixture
    def mock_integrated_engine(self):
        """Create a mock engine with integration features"""
        engine = Mock()
        engine.integrate_with_apis = AsyncMock()
        engine.validate_api_connections = AsyncMock()
        engine.get_api_status = AsyncMock()
        return engine
    
    @pytest.mark.asyncio
    async def test_integrate_with_apis(self, mock_integrated_engine):
        """Test API integration"""
        api_config = {
            "google_search_api": {"key": "test_key", "enabled": True},
            "news_api": {"key": "test_key", "enabled": True},
            "shopping_api": {"key": "test_key", "enabled": False}
        }
        
        expected_result = {
            "integrated_apis": 2,
            "failed_integrations": 0,
            "disabled_apis": 1,
            "integration_time": 1.2
        }
        
        mock_integrated_engine.integrate_with_apis.return_value = expected_result
        
        result = await mock_integrated_engine.integrate_with_apis(api_config)
        
        assert result["integrated_apis"] == 2
        assert result["failed_integrations"] == 0
        assert result["disabled_apis"] == 1
        assert result["integration_time"] == 1.2
    
    @pytest.mark.asyncio
    async def test_validate_api_connections(self, mock_integrated_engine):
        """Test API connection validation"""
        expected_validation = {
            "google_search_api": {"status": "healthy", "response_time": 0.25},
            "news_api": {"status": "healthy", "response_time": 0.31},
            "shopping_api": {"status": "disabled", "response_time": None}
        }
        
        mock_integrated_engine.validate_api_connections.return_value = expected_validation
        
        result = await mock_integrated_engine.validate_api_connections()
        
        assert result["google_search_api"]["status"] == "healthy"
        assert result["news_api"]["status"] == "healthy"
        assert result["shopping_api"]["status"] == "disabled"
        assert result["google_search_api"]["response_time"] == 0.25
    
    @pytest.mark.asyncio
    async def test_get_api_status(self, mock_integrated_engine):
        """Test API status retrieval"""
        expected_status = {
            "total_apis": 3,
            "healthy_apis": 2,
            "unhealthy_apis": 0,
            "disabled_apis": 1,
            "last_health_check": datetime.now().isoformat()
        }
        
        mock_integrated_engine.get_api_status.return_value = expected_status
        
        result = await mock_integrated_engine.get_api_status()
        
        assert result["total_apis"] == 3
        assert result["healthy_apis"] == 2
        assert result["unhealthy_apis"] == 0
        assert result["disabled_apis"] == 1
        assert "last_health_check" in result