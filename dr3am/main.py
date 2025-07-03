"""Main application entry point for dr3am"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from dr3am.utils.config import get_settings
from dr3am.utils.logging import setup_logging, get_logger
from dr3am.models.database import init_database
from dr3am.auth.routes import router as auth_router
from dr3am.monitoring.middleware import LoggingMiddleware, MetricsMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    settings = get_settings()
    logger = get_logger("dr3am.main")
    
    logger.info("Starting dr3am application...")
    
    # Initialize database
    try:
        db_manager = init_database(settings.get_database_url(), echo=settings.database.echo)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Initialize other services here
    logger.info("dr3am application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down dr3am application...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()
    
    # Set up logging
    setup_logging(settings.logging)
    logger = get_logger("dr3am.main")
    
    # Create FastAPI app
    app = FastAPI(
        title="dr3am - Autonomous Agent MCP Server",
        description="""
        Transform reactive chatbots into proactive research assistants.
        
        ## Features
        
        * **Semantic Interest Analysis** - Analyze conversation patterns to identify user interests
        * **Autonomous Research** - Generate and execute research plans when agents are idle  
        * **MCP Protocol** - Standard Model Context Protocol for universal agent compatibility
        * **Real API Integration** - Connect to Google Search, News APIs, and more
        * **Multi-user Support** - Secure user management and authentication
        * **Performance Monitoring** - Comprehensive metrics and health checks
        
        ## Authentication
        
        Most endpoints require authentication. Use the `/auth/login` endpoint to obtain an access token,
        then include it in the Authorization header: `Authorization: Bearer <token>`
        
        ## Rate Limiting
        
        API calls are rate limited to ensure fair usage and cost control. Current limits:
        - 60 requests per minute per user
        - 1000 requests per day per user
        - Daily API budget limits apply to external service calls
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=settings.security.cors_credentials,
        allow_methods=settings.security.cors_methods,
        allow_headers=settings.security.cors_headers,
    )
    
    # Add monitoring middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    
    # Include routers
    app.include_router(auth_router, prefix="/api")
    
    # Add core MCP endpoints
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with basic information"""
        return {
            "name": "dr3am",
            "version": "1.0.0",
            "description": "Autonomous Agent MCP Server",
            "docs_url": "/docs",
            "health_url": "/health"
        }
    
    @app.get("/health", tags=["monitoring"])
    async def health_check():
        """Health check endpoint"""
        from datetime import datetime
        import psutil
        import os
        
        try:
            # Basic health check
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": "1.0.0",
                "environment": settings.environment,
                "uptime": psutil.boot_time(),
                "components": {
                    "database": "healthy",  # TODO: Add actual DB health check
                    "redis": "healthy",     # TODO: Add actual Redis health check
                    "api_services": "healthy"  # TODO: Add actual API health checks
                },
                "system": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                }
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "error": str(e)
                }
            )
    
    @app.get("/metrics", tags=["monitoring"])
    async def metrics():
        """Prometheus metrics endpoint"""
        # TODO: Implement actual metrics collection
        return {"message": "Metrics endpoint - TODO: Implement Prometheus metrics"}
    
    # Customize OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="dr3am API",
            version="1.0.0",
            description=app.description,
            routes=app.routes,
        )
        
        # Add security schemas
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
        
        # Add global security requirement
        openapi_schema["security"] = [{"BearerAuth": []}]
        
        # Add custom schemas
        openapi_schema["components"]["schemas"].update({
            "ConversationMessage": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Message content"},
                    "timestamp": {"type": "string", "format": "date-time", "description": "Message timestamp"},
                    "role": {"type": "string", "enum": ["user", "assistant"], "description": "Message role"},
                    "conversation_id": {"type": "string", "description": "Conversation identifier"},
                    "metadata": {"type": "object", "description": "Additional metadata"}
                },
                "required": ["content", "timestamp", "role", "conversation_id"]
            },
            "DetectedInterest": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Interest topic"},
                    "interest_type": {"type": "string", "enum": ["hobby", "problem", "learning", "health", "goal", "preference", "concern"]},
                    "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "engagement_level": {"type": "integer", "minimum": 1, "maximum": 4},
                    "mention_count": {"type": "integer", "minimum": 1}
                }
            },
            "ResearchOpportunity": {
                "type": "object",
                "properties": {
                    "interest_topic": {"type": "string"},
                    "research_type": {"type": "string", "enum": ["find_resources", "deep_dive", "monitor", "comparison"]},
                    "priority_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "expected_value": {"type": "string", "enum": ["high", "medium", "low"]},
                    "urgency": {"type": "string", "enum": ["urgent", "high", "medium", "low"]}
                }
            }
        })
        
        # Add examples
        openapi_schema["components"]["examples"] = {
            "ConversationExample": {
                "summary": "Sample conversation message",
                "value": {
                    "content": "I'm interested in learning Python programming",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "role": "user",
                    "conversation_id": "conv_123"
                }
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    return app


# Create the app instance
app = create_app()


def main():
    """Main entry point for running the application"""
    settings = get_settings()
    
    uvicorn.run(
        "dr3am.main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        reload=settings.server.reload,
        log_level=settings.logging.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()