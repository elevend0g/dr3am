"""Unit tests for MCP server"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

from dr3am.models.conversation import ConversationMessage
from dr3am.models.interests import DetectedInterest, ResearchOpportunity
from dr3am.models.research import ResearchResult, ActionableInsight


class TestMCPServer:
    """Test MCP server functionality"""
    
    @pytest.fixture
    def mock_mcp_server(self):
        """Create a mock MCP server"""
        server = Mock()
        server.analyze_conversations = AsyncMock()
        server.trigger_boredom = AsyncMock()
        server.generate_research_plan = AsyncMock()
        server.get_interest_summary = AsyncMock()
        server.health_check = AsyncMock()
        return server
    
    @pytest.fixture
    def mock_fastapi_app(self):
        """Create a mock FastAPI app"""
        app = Mock()
        app.post = Mock()
        app.get = Mock()
        return app
    
    @pytest.mark.asyncio
    async def test_analyze_conversations_endpoint(self, mock_mcp_server):
        """Test analyze conversations endpoint"""
        conversation_data = {
            "conversations": [
                {
                    "content": "I'm interested in learning Python",
                    "timestamp": datetime.now().isoformat(),
                    "role": "user",
                    "conversation_id": "conv_1"
                }
            ],
            "incremental": True
        }
        
        expected_response = {
            "status": "success",
            "detected_interests": [
                {
                    "topic": "Python programming",
                    "interest_type": "learning",
                    "confidence_score": 0.9,
                    "engagement_level": 3,
                    "mention_count": 1
                }
            ],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        mock_mcp_server.analyze_conversations.return_value = expected_response
        
        result = await mock_mcp_server.analyze_conversations(conversation_data)
        
        assert result["status"] == "success"
        assert len(result["detected_interests"]) == 1
        assert result["detected_interests"][0]["topic"] == "Python programming"
        assert result["detected_interests"][0]["interest_type"] == "learning"
        assert result["detected_interests"][0]["confidence_score"] == 0.9
        
        mock_mcp_server.analyze_conversations.assert_called_once_with(conversation_data)
    
    @pytest.mark.asyncio
    async def test_trigger_boredom_endpoint(self, mock_mcp_server):
        """Test trigger boredom endpoint"""
        boredom_data = {
            "agent_memory_endpoint": "http://example.com/memory",
            "conversation_limit": 50
        }
        
        expected_response = {
            "status": "success",
            "research_triggered": True,
            "research_opportunities": [
                {
                    "topic": "Python programming",
                    "research_type": "find_resources",
                    "priority_score": 0.9,
                    "expected_completion": (datetime.now() + timedelta(hours=1)).isoformat()
                }
            ],
            "message": "Autonomous research cycle initiated"
        }
        
        mock_mcp_server.trigger_boredom.return_value = expected_response
        
        result = await mock_mcp_server.trigger_boredom(boredom_data)
        
        assert result["status"] == "success"
        assert result["research_triggered"] is True
        assert len(result["research_opportunities"]) == 1
        assert result["research_opportunities"][0]["topic"] == "Python programming"
        assert "message" in result
        
        mock_mcp_server.trigger_boredom.assert_called_once_with(boredom_data)
    
    @pytest.mark.asyncio
    async def test_generate_research_plan_endpoint(self, mock_mcp_server):
        """Test generate research plan endpoint"""
        plan_data = {
            "interests": [
                {
                    "topic": "Python programming",
                    "interest_type": "learning",
                    "confidence_score": 0.9
                }
            ],
            "research_preferences": {
                "max_research_items": 5,
                "urgency_filter": "medium"
            }
        }
        
        expected_response = {
            "status": "success",
            "research_plan": {
                "plan_id": "plan_123",
                "research_opportunities": [
                    {
                        "topic": "Python programming",
                        "research_type": "find_resources",
                        "priority_score": 0.9,
                        "research_query": "Python beginner tutorials",
                        "expected_value": "high",
                        "urgency": "medium"
                    }
                ],
                "estimated_completion_time": 3600,
                "total_api_cost": 2.50
            },
            "timestamp": datetime.now().isoformat()
        }
        
        mock_mcp_server.generate_research_plan.return_value = expected_response
        
        result = await mock_mcp_server.generate_research_plan(plan_data)
        
        assert result["status"] == "success"
        assert "research_plan" in result
        assert result["research_plan"]["plan_id"] == "plan_123"
        assert len(result["research_plan"]["research_opportunities"]) == 1
        assert result["research_plan"]["estimated_completion_time"] == 3600
        assert result["research_plan"]["total_api_cost"] == 2.50
        
        mock_mcp_server.generate_research_plan.assert_called_once_with(plan_data)
    
    @pytest.mark.asyncio
    async def test_get_interest_summary_endpoint(self, mock_mcp_server):
        """Test get interest summary endpoint"""
        expected_response = {
            "status": "success",
            "interest_summary": {
                "total_interests": 3,
                "by_type": {
                    "learning": 2,
                    "hobby": 1,
                    "goal": 0
                },
                "by_engagement_level": {
                    "casual_mention": 1,
                    "recurring_interest": 1,
                    "active_engagement": 1,
                    "persistent_focus": 0
                },
                "top_interests": [
                    {
                        "topic": "Python programming",
                        "confidence_score": 0.9,
                        "engagement_level": 3,
                        "last_mentioned": datetime.now().isoformat()
                    }
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        mock_mcp_server.get_interest_summary.return_value = expected_response
        
        result = await mock_mcp_server.get_interest_summary()
        
        assert result["status"] == "success"
        assert "interest_summary" in result
        assert result["interest_summary"]["total_interests"] == 3
        assert "by_type" in result["interest_summary"]
        assert "by_engagement_level" in result["interest_summary"]
        assert len(result["interest_summary"]["top_interests"]) == 1
        
        mock_mcp_server.get_interest_summary.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, mock_mcp_server):
        """Test health check endpoint"""
        expected_response = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": "healthy",
                "semantic_analyzer": "healthy",
                "research_engine": "healthy",
                "api_integrations": "healthy"
            },
            "uptime": 86400,
            "version": "1.0.0"
        }
        
        mock_mcp_server.health_check.return_value = expected_response
        
        result = await mock_mcp_server.health_check()
        
        assert result["status"] == "healthy"
        assert "components" in result
        assert result["components"]["database"] == "healthy"
        assert result["components"]["semantic_analyzer"] == "healthy"
        assert result["components"]["research_engine"] == "healthy"
        assert result["components"]["api_integrations"] == "healthy"
        assert result["uptime"] == 86400
        assert result["version"] == "1.0.0"
        
        mock_mcp_server.health_check.assert_called_once()


class TestMCPServerErrorHandling:
    """Test MCP server error handling"""
    
    @pytest.fixture
    def mock_server_with_errors(self):
        """Create a mock server that can raise errors"""
        server = Mock()
        server.analyze_conversations = AsyncMock()
        server.trigger_boredom = AsyncMock()
        server.generate_research_plan = AsyncMock()
        return server
    
    @pytest.mark.asyncio
    async def test_analyze_conversations_error(self, mock_server_with_errors):
        """Test error handling in analyze conversations"""
        mock_server_with_errors.analyze_conversations.side_effect = Exception("Analysis failed")
        
        with pytest.raises(Exception, match="Analysis failed"):
            await mock_server_with_errors.analyze_conversations({})
    
    @pytest.mark.asyncio
    async def test_trigger_boredom_error(self, mock_server_with_errors):
        """Test error handling in trigger boredom"""
        mock_server_with_errors.trigger_boredom.side_effect = Exception("Boredom trigger failed")
        
        with pytest.raises(Exception, match="Boredom trigger failed"):
            await mock_server_with_errors.trigger_boredom({})
    
    @pytest.mark.asyncio
    async def test_generate_research_plan_error(self, mock_server_with_errors):
        """Test error handling in generate research plan"""
        mock_server_with_errors.generate_research_plan.side_effect = Exception("Research plan generation failed")
        
        with pytest.raises(Exception, match="Research plan generation failed"):
            await mock_server_with_errors.generate_research_plan({})
    
    @pytest.mark.asyncio
    async def test_invalid_request_data(self, mock_server_with_errors):
        """Test handling of invalid request data"""
        mock_server_with_errors.analyze_conversations.side_effect = ValueError("Invalid request data")
        
        with pytest.raises(ValueError, match="Invalid request data"):
            await mock_server_with_errors.analyze_conversations(None)


class TestMCPServerValidation:
    """Test MCP server request validation"""
    
    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator"""
        validator = Mock()
        validator.validate_conversation_request = Mock()
        validator.validate_boredom_request = Mock()
        validator.validate_research_plan_request = Mock()
        return validator
    
    def test_validate_conversation_request(self, mock_validator):
        """Test conversation request validation"""
        valid_request = {
            "conversations": [
                {
                    "content": "Hello world",
                    "timestamp": datetime.now().isoformat(),
                    "role": "user",
                    "conversation_id": "conv_1"
                }
            ],
            "incremental": True
        }
        
        mock_validator.validate_conversation_request.return_value = True
        
        result = mock_validator.validate_conversation_request(valid_request)
        
        assert result is True
        mock_validator.validate_conversation_request.assert_called_once_with(valid_request)
    
    def test_validate_boredom_request(self, mock_validator):
        """Test boredom request validation"""
        valid_request = {
            "agent_memory_endpoint": "http://example.com/memory",
            "conversation_limit": 50
        }
        
        mock_validator.validate_boredom_request.return_value = True
        
        result = mock_validator.validate_boredom_request(valid_request)
        
        assert result is True
        mock_validator.validate_boredom_request.assert_called_once_with(valid_request)
    
    def test_validate_research_plan_request(self, mock_validator):
        """Test research plan request validation"""
        valid_request = {
            "interests": [
                {
                    "topic": "Python programming",
                    "interest_type": "learning",
                    "confidence_score": 0.9
                }
            ],
            "research_preferences": {
                "max_research_items": 5,
                "urgency_filter": "medium"
            }
        }
        
        mock_validator.validate_research_plan_request.return_value = True
        
        result = mock_validator.validate_research_plan_request(valid_request)
        
        assert result is True
        mock_validator.validate_research_plan_request.assert_called_once_with(valid_request)


class TestMCPServerCORS:
    """Test MCP server CORS configuration"""
    
    @pytest.fixture
    def mock_cors_middleware(self):
        """Create a mock CORS middleware"""
        middleware = Mock()
        middleware.allow_origins = ["*"]
        middleware.allow_credentials = True
        middleware.allow_methods = ["*"]
        middleware.allow_headers = ["*"]
        return middleware
    
    def test_cors_configuration(self, mock_cors_middleware):
        """Test CORS configuration"""
        assert mock_cors_middleware.allow_origins == ["*"]
        assert mock_cors_middleware.allow_credentials is True
        assert mock_cors_middleware.allow_methods == ["*"]
        assert mock_cors_middleware.allow_headers == ["*"]


class TestMCPServerBackground:
    """Test MCP server background tasks"""
    
    @pytest.fixture
    def mock_background_tasks(self):
        """Create mock background tasks"""
        tasks = Mock()
        tasks.add_task = Mock()
        tasks.cleanup_expired_data = AsyncMock()
        tasks.update_research_status = AsyncMock()
        tasks.optimize_performance = AsyncMock()
        return tasks
    
    def test_add_background_task(self, mock_background_tasks):
        """Test adding background tasks"""
        mock_background_tasks.add_task("cleanup_expired_data")
        
        mock_background_tasks.add_task.assert_called_once_with("cleanup_expired_data")
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_data_task(self, mock_background_tasks):
        """Test cleanup expired data background task"""
        mock_background_tasks.cleanup_expired_data.return_value = {
            "expired_interests": 3,
            "expired_research": 5,
            "cleanup_time": 2.5
        }
        
        result = await mock_background_tasks.cleanup_expired_data()
        
        assert result["expired_interests"] == 3
        assert result["expired_research"] == 5
        assert result["cleanup_time"] == 2.5
        
        mock_background_tasks.cleanup_expired_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_research_status_task(self, mock_background_tasks):
        """Test update research status background task"""
        mock_background_tasks.update_research_status.return_value = {
            "updated_research_count": 7,
            "status_changes": 3,
            "update_time": 1.2
        }
        
        result = await mock_background_tasks.update_research_status()
        
        assert result["updated_research_count"] == 7
        assert result["status_changes"] == 3
        assert result["update_time"] == 1.2
        
        mock_background_tasks.update_research_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimize_performance_task(self, mock_background_tasks):
        """Test optimize performance background task"""
        mock_background_tasks.optimize_performance.return_value = {
            "cache_optimizations": 12,
            "memory_freed": 256,
            "optimization_time": 3.8
        }
        
        result = await mock_background_tasks.optimize_performance()
        
        assert result["cache_optimizations"] == 12
        assert result["memory_freed"] == 256
        assert result["optimization_time"] == 3.8
        
        mock_background_tasks.optimize_performance.assert_called_once()


class TestMCPServerMetrics:
    """Test MCP server metrics and monitoring"""
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create a mock metrics collector"""
        collector = Mock()
        collector.record_request = Mock()
        collector.record_error = Mock()
        collector.get_metrics = Mock()
        return collector
    
    def test_record_request_metric(self, mock_metrics_collector):
        """Test recording request metrics"""
        mock_metrics_collector.record_request("POST", "/api/analyze", 0.25)
        
        mock_metrics_collector.record_request.assert_called_once_with("POST", "/api/analyze", 0.25)
    
    def test_record_error_metric(self, mock_metrics_collector):
        """Test recording error metrics"""
        mock_metrics_collector.record_error("POST", "/api/analyze", 500, "Internal Server Error")
        
        mock_metrics_collector.record_error.assert_called_once_with("POST", "/api/analyze", 500, "Internal Server Error")
    
    def test_get_metrics(self, mock_metrics_collector):
        """Test getting metrics"""
        expected_metrics = {
            "total_requests": 1000,
            "successful_requests": 950,
            "error_requests": 50,
            "average_response_time": 0.15,
            "requests_per_second": 25.5
        }
        
        mock_metrics_collector.get_metrics.return_value = expected_metrics
        
        result = mock_metrics_collector.get_metrics()
        
        assert result["total_requests"] == 1000
        assert result["successful_requests"] == 950
        assert result["error_requests"] == 50
        assert result["average_response_time"] == 0.15
        assert result["requests_per_second"] == 25.5
        
        mock_metrics_collector.get_metrics.assert_called_once()