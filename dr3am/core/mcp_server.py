from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from enum import Enum

# Import our semantic analyzer
from semantic_analyzer import (
    SemanticInterestAnalyzer, 
    ConversationMessage, 
    DetectedInterest, 
    ResearchOpportunity,
    InterestType,
    EngagementLevel
)


# MCP Protocol Models
class MCPRequest(BaseModel):
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))


class MCPResponse(BaseModel):
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None
    id: Optional[str] = None


class MCPError(BaseModel):
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


# dr3am specific request/response models
class ConversationAnalysisRequest(BaseModel):
    conversations: List[Dict[str, Any]]
    incremental: bool = True
    analysis_config: Optional[Dict[str, Any]] = None


class BoredomTriggerRequest(BaseModel):
    agent_memory_endpoint: Optional[str] = None
    conversation_limit: int = 50
    force_analysis: bool = False


class ResearchPlanRequest(BaseModel):
    interests: Optional[List[Dict[str, Any]]] = None
    max_opportunities: int = 5
    priority_threshold: float = 0.5


class InterestSummaryRequest(BaseModel):
    topic_filter: Optional[str] = None
    min_confidence: float = 0.6
    include_research_potential: bool = True


class ConfigurationRequest(BaseModel):
    analysis_window_days: Optional[int] = None
    min_mentions_for_interest: Optional[int] = None
    confidence_threshold: Optional[float] = None
    max_interests_per_analysis: Optional[int] = None


# Background task status tracking
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundTask(BaseModel):
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Dr3amMCPServer:
    """
    MCP Server implementation for dr3am autonomous agent capabilities.
    Exposes semantic interest analysis and research planning as standardized MCP tools.
    """
    
    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        self.app = FastAPI(
            title="dr3am MCP Server",
            description="Autonomous agent capabilities for self-improvement and proactive research",
            version="0.1.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize semantic analyzer
        self.analyzer = SemanticInterestAnalyzer(llm_client=llm_client, config=config)
        
        # Background task tracking
        self.background_tasks: Dict[str, BackgroundTask] = {}
        
        # MCP tool definitions
        self.tools = self._define_mcp_tools()
        
        # Setup routes
        self._setup_routes()
    
    def _define_mcp_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define available MCP tools for agent registration"""
        return {
            "analyze_conversations": {
                "description": "Analyze conversation history for user interests and patterns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "conversations": {
                            "type": "array",
                            "description": "List of conversation messages to analyze"
                        },
                        "incremental": {
                            "type": "boolean", 
                            "description": "Only analyze new conversations",
                            "default": True
                        }
                    },
                    "required": ["conversations"]
                }
            },
            "trigger_boredom": {
                "description": "Trigger autonomous research cycle when agent is idle",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_memory_endpoint": {
                            "type": "string",
                            "description": "Optional endpoint to fetch agent memory"
                        },
                        "conversation_limit": {
                            "type": "integer",
                            "description": "Number of recent conversations to analyze",
                            "default": 50
                        }
                    }
                }
            },
            "generate_research_plan": {
                "description": "Generate research opportunities from detected interests",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "interests": {
                            "type": "array",
                            "description": "Optional list of interests to research"
                        },
                        "max_opportunities": {
                            "type": "integer",
                            "description": "Maximum research opportunities to generate",
                            "default": 5
                        }
                    }
                }
            },
            "get_interest_summary": {
                "description": "Get summary of currently tracked user interests",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_filter": {
                            "type": "string",
                            "description": "Filter interests by topic keyword"
                        },
                        "min_confidence": {
                            "type": "number",
                            "description": "Minimum confidence threshold",
                            "default": 0.6
                        }
                    }
                }
            },
            "configure_analyzer": {
                "description": "Update analyzer configuration settings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_window_days": {"type": "integer"},
                        "min_mentions_for_interest": {"type": "integer"},
                        "confidence_threshold": {"type": "number"},
                        "max_interests_per_analysis": {"type": "integer"}
                    }
                }
            }
        }
    
    def _setup_routes(self):
        """Setup FastAPI routes for MCP protocol"""
        
        @self.app.get("/")
        async def root():
            return {
                "name": "dr3am",
                "version": "0.1.0",
                "description": "Autonomous agent capabilities MCP server",
                "protocol": "mcp/1.0",
                "tools": list(self.tools.keys())
            }
        
        @self.app.get("/tools")
        async def list_tools():
            """List available MCP tools"""
            return {"tools": self.tools}
        
        @self.app.post("/tools/call")
        async def call_tool(request: MCPRequest):
            """Main MCP tool calling endpoint"""
            
            try:
                method = request.method
                params = request.params
                
                if method == "analyze_conversations":
                    result = await self._analyze_conversations(params)
                elif method == "trigger_boredom":
                    result = await self._trigger_boredom(params)
                elif method == "generate_research_plan":
                    result = await self._generate_research_plan(params)
                elif method == "get_interest_summary":
                    result = await self._get_interest_summary(params)
                elif method == "configure_analyzer":
                    result = await self._configure_analyzer(params)
                else:
                    raise HTTPException(status_code=404, detail=f"Tool '{method}' not found")
                
                return MCPResponse(result=result, id=request.id)
                
            except Exception as e:
                error = MCPError(
                    code=-32603,
                    message=str(e),
                    data={"method": request.method}
                )
                return MCPResponse(error=error.dict(), id=request.id)
        
        @self.app.post("/boredom/trigger")
        async def trigger_boredom_endpoint(
            request: BoredomTriggerRequest, 
            background_tasks: BackgroundTasks
        ):
            """Dedicated endpoint for boredom trigger with background processing"""
            
            task_id = str(uuid.uuid4())
            task = BackgroundTask(
                task_id=task_id,
                task_type="boredom_cycle",
                status=TaskStatus.PENDING,
                created_at=datetime.now()
            )
            
            self.background_tasks[task_id] = task
            
            # Start background boredom cycle
            background_tasks.add_task(
                self._execute_boredom_cycle, 
                task_id, 
                request.dict()
            )
            
            return {"task_id": task_id, "status": "started"}
        
        @self.app.get("/tasks/{task_id}")
        async def get_task_status(task_id: str):
            """Get status of background task"""
            if task_id not in self.background_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return self.background_tasks[task_id].dict()
        
        @self.app.get("/interests")
        async def get_current_interests():
            """Get all currently tracked interests"""
            interests = list(self.analyzer.interest_history.values())
            return {
                "interests": [self._serialize_interest(interest) for interest in interests],
                "total_count": len(interests),
                "last_updated": datetime.now().isoformat()
            }
        
        @self.app.post("/interests/reset")
        async def reset_interests():
            """Reset all tracked interests (for testing/debugging)"""
            self.analyzer.interest_history.clear()
            self.analyzer.analyzed_conversations.clear()
            return {"message": "Interest history reset successfully"}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "tracked_interests": len(self.analyzer.interest_history),
                "analyzed_conversations": len(self.analyzer.analyzed_conversations)
            }
    
    async def _analyze_conversations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversations for interests and patterns"""
        
        # Convert conversation data to ConversationMessage objects
        conversations = []
        for conv_data in params.get("conversations", []):
            conversations.append(ConversationMessage(
                content=conv_data["content"],
                timestamp=datetime.fromisoformat(conv_data["timestamp"]),
                role=conv_data.get("role", "user"),
                conversation_id=conv_data.get("conversation_id", str(uuid.uuid4())),
                metadata=conv_data.get("metadata", {})
            ))
        
        # Perform analysis
        incremental = params.get("incremental", True)
        interests = self.analyzer.analyze_conversations(conversations, incremental)
        
        return {
            "interests": [self._serialize_interest(interest) for interest in interests],
            "analysis_timestamp": datetime.now().isoformat(),
            "conversations_analyzed": len(conversations),
            "new_interests_found": len([i for i in interests 
                                       if i.first_mentioned >= datetime.now() - timedelta(hours=1)])
        }
    
    async def _trigger_boredom(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger boredom cycle - analyze recent conversations and generate research plan"""
        
        # This would typically fetch conversations from the agent's memory
        # For now, we'll work with whatever conversations are already analyzed
        interests = list(self.analyzer.interest_history.values())
        
        if not interests:
            return {
                "message": "No interests detected. Agent needs more conversation history.",
                "action": "wait_for_conversations"
            }
        
        # Generate research opportunities
        opportunities = self.analyzer.generate_research_opportunities(interests)
        
        # Select top priority opportunity for immediate action
        if opportunities:
            top_opportunity = max(opportunities, key=lambda x: x.priority_score)
            
            return {
                "boredom_triggered": True,
                "action": "research_suggested",
                "immediate_research": {
                    "topic": top_opportunity.interest_topic,
                    "question": top_opportunity.research_question,
                    "priority": top_opportunity.priority_score,
                    "suggested_tools": top_opportunity.suggested_tools,
                    "reasoning": top_opportunity.reasoning
                },
                "total_opportunities": len(opportunities),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "boredom_triggered": True,
                "action": "no_research_needed", 
                "message": "No high-priority research opportunities identified",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _generate_research_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research plan from interests"""
        
        interests_data = params.get("interests")
        if interests_data:
            # Convert provided interests
            interests = [self._deserialize_interest(data) for data in interests_data]
        else:
            # Use current tracked interests
            interests = list(self.analyzer.interest_history.values())
        
        max_opportunities = params.get("max_opportunities", 5)
        opportunities = self.analyzer.generate_research_opportunities(interests)
        
        # Limit and sort by priority
        top_opportunities = sorted(
            opportunities, 
            key=lambda x: x.priority_score, 
            reverse=True
        )[:max_opportunities]
        
        return {
            "research_plan": [self._serialize_opportunity(opp) for opp in top_opportunities],
            "total_interests_analyzed": len(interests),
            "plan_generated_at": datetime.now().isoformat()
        }
    
    async def _get_interest_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of tracked interests"""
        
        interests = list(self.analyzer.interest_history.values())
        
        # Apply filters
        topic_filter = params.get("topic_filter")
        min_confidence = params.get("min_confidence", 0.6)
        
        filtered_interests = []
        for interest in interests:
            if interest.confidence_score < min_confidence:
                continue
            
            if topic_filter and topic_filter.lower() not in interest.topic.lower():
                continue
                
            filtered_interests.append(interest)
        
        # Group by interest type
        by_type = {}
        for interest in filtered_interests:
            interest_type = interest.interest_type.value
            if interest_type not in by_type:
                by_type[interest_type] = []
            by_type[interest_type].append(self._serialize_interest(interest))
        
        return {
            "summary": {
                "total_interests": len(filtered_interests),
                "by_type": by_type,
                "most_engaged": [
                    self._serialize_interest(interest) 
                    for interest in sorted(filtered_interests, 
                                         key=lambda x: x.engagement_level.value, 
                                         reverse=True)[:3]
                ],
                "most_recent": [
                    self._serialize_interest(interest)
                    for interest in sorted(filtered_interests,
                                         key=lambda x: x.last_mentioned,
                                         reverse=True)[:3]
                ]
            },
            "generated_at": datetime.now().isoformat()
        }
    
    async def _configure_analyzer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update analyzer configuration"""
        
        valid_configs = [
            "analysis_window_days", 
            "min_mentions_for_interest", 
            "confidence_threshold",
            "max_interests_per_analysis"
        ]
        
        updated_configs = {}
        for key, value in params.items():
            if key in valid_configs:
                self.analyzer.config[key] = value
                updated_configs[key] = value
        
        return {
            "updated_configs": updated_configs,
            "current_config": self.analyzer.config,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_boredom_cycle(self, task_id: str, params: Dict[str, Any]):
        """Execute full boredom cycle in background"""
        
        task = self.background_tasks[task_id]
        task.status = TaskStatus.RUNNING
        
        try:
            # Simulate autonomous research cycle
            await asyncio.sleep(1)  # Simulate processing time
            
            # Analyze current interests
            interests = list(self.analyzer.interest_history.values())
            
            # Generate research opportunities  
            opportunities = self.analyzer.generate_research_opportunities(interests)
            
            # Execute top research task (would integrate with actual research tools)
            if opportunities:
                top_opp = max(opportunities, key=lambda x: x.priority_score)
                
                # Simulate research execution
                research_results = await self._simulate_research_execution(top_opp)
                
                task.result = {
                    "research_completed": True,
                    "topic": top_opp.interest_topic,
                    "findings": research_results,
                    "opportunities_identified": len(opportunities)
                }
            else:
                task.result = {
                    "research_completed": False,
                    "reason": "No research opportunities identified"
                }
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
    
    async def _simulate_research_execution(self, opportunity: ResearchOpportunity) -> Dict[str, Any]:
        """Simulate research execution (would integrate with actual tools)"""
        
        await asyncio.sleep(2)  # Simulate research time
        
        return {
            "summary": f"Research completed for {opportunity.interest_topic}",
            "sources_found": 5,
            "key_insights": [
                "Found 3 new resources relevant to user interests",
                "Identified upcoming event that might interest user", 
                "Discovered cost-saving opportunity"
            ],
            "next_actions": [
                "Monitor for sales/discounts",
                "Check for new tutorials/resources weekly",
                "Alert user about relevant events"
            ]
        }
    
    def _serialize_interest(self, interest: DetectedInterest) -> Dict[str, Any]:
        """Convert DetectedInterest to JSON-serializable dict"""
        return {
            "topic": interest.topic,
            "type": interest.interest_type.value,
            "engagement_level": interest.engagement_level.value,
            "confidence_score": interest.confidence_score,
            "first_mentioned": interest.first_mentioned.isoformat(),
            "last_mentioned": interest.last_mentioned.isoformat(),
            "mention_count": interest.mention_count,
            "key_phrases": interest.key_phrases,
            "context_summary": interest.context_summary,
            "research_potential": interest.research_potential,
            "related_topics": interest.related_topics
        }
    
    def _serialize_opportunity(self, opportunity: ResearchOpportunity) -> Dict[str, Any]:
        """Convert ResearchOpportunity to JSON-serializable dict"""
        return {
            "interest_topic": opportunity.interest_topic,
            "research_question": opportunity.research_question,
            "priority_score": opportunity.priority_score,
            "estimated_effort": opportunity.estimated_effort,
            "suggested_tools": opportunity.suggested_tools,
            "research_type": opportunity.research_type,
            "reasoning": opportunity.reasoning
        }
    
    def _deserialize_interest(self, data: Dict[str, Any]) -> DetectedInterest:
        """Convert dict back to DetectedInterest object"""
        return DetectedInterest(
            topic=data["topic"],
            interest_type=InterestType(data["type"]),
            engagement_level=EngagementLevel(data["engagement_level"]),
            confidence_score=data["confidence_score"],
            first_mentioned=datetime.fromisoformat(data["first_mentioned"]),
            last_mentioned=datetime.fromisoformat(data["last_mentioned"]),
            mention_count=data["mention_count"],
            key_phrases=data["key_phrases"],
            context_summary=data["context_summary"],
            research_potential=data["research_potential"],
            related_topics=data.get("related_topics", [])
        )


# FastAPI app factory
def create_dr3am_server(llm_client=None, config: Dict[str, Any] = None) -> FastAPI:
    """Create and configure dr3am MCP server"""
    server = Dr3amMCPServer(llm_client=llm_client, config=config)
    return server.app


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Create server instance
    app = create_dr3am_server()
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True  # For development
    )
