# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dr3am** is an autonomous agent MCP (Model Context Protocol) server that transforms reactive chatbots into proactive research assistants. It analyzes conversation patterns to identify user interests and autonomously generates research opportunities when agents are idle.

## Key Commands

### Development
```bash
# Start the dr3am MCP server
python dr3am_mcp_server.py

# Start the integrated system
python integrated_dr3am_system.py

# Run test client
python dr3am_test_client.py

# Install dependencies
pip install fastapi uvicorn aiohttp pydantic
```

### Server Configuration
- Default server runs on `http://localhost:8000`
- Configure via environment variables: `DR3AM_HOST`, `DR3AM_PORT`
- Analysis settings: `ANALYSIS_WINDOW_DAYS`, `MIN_MENTIONS`, `CONFIDENCE_THRESHOLD`

## Core Architecture

### Main Components

1. **Semantic Interest Analyzer** (`semantic_analyzer.py`)
   - LLM-powered conversation analysis
   - Detects user interests from conversation patterns
   - Tracks engagement levels and interest types (hobby, problem, learning, health, goal, preference, concern)

2. **MCP Server** (`dr3am_mcp_server.py`)
   - FastAPI-based server implementing MCP protocol
   - Provides tools: `analyze_conversations`, `trigger_boredom`, `generate_research_plan`
   - Handles CORS and background task processing

3. **Research Execution Engine** (`research_execution_engine.py`)
   - Orchestrates autonomous research based on detected interests
   - Manages research results and actionable insights
   - Handles web search integration and API monitoring

4. **Integrated System** (`integrated_dr3am_system.py`)
   - Combines semantic analysis with research execution
   - Manages system state and performance tracking
   - Provides complete autonomous research lifecycle

### Data Models

- `ConversationMessage`: Standardized message format with content and timestamp
- `DetectedInterest`: Interest with type, engagement level, and confidence score
- `ResearchOpportunity`: Actionable research tasks with priority scoring
- `ResearchResult`: Container for research findings with insights and sources

## Key Features

### Interest Detection
- Analyzes conversation history to identify patterns
- Scores interests by engagement level (casual mention → persistent focus)
- Supports configurable confidence thresholds and mention minimums

### Autonomous Research
- Triggers research cycles when agents are idle ("boredom loop")
- Generates research plans based on detected interests
- Provides proactive insights rather than waiting for user requests

### MCP Integration
- Standard MCP protocol for universal agent compatibility
- RESTful endpoints for HTTP integration
- Background task processing for autonomous cycles

## Integration Examples

### Basic Usage
```python
from integration_examples import Dr3amClient

async with Dr3amClient() as dr3am:
    # Analyze conversations
    result = await dr3am.analyze_conversations(conversation_history)
    
    # Trigger autonomous research
    research = await dr3am.trigger_boredom()
    
    # Get current interests
    interests = await dr3am.get_interest_summary()
```

### Framework Integration
- LangChain: Use as custom tool with `@tool` decorator
- AutoGen: Add as tool in agent configuration
- Custom agents: Integrate via `Dr3amClient` class

## Configuration

### Analysis Parameters
- `analysis_window_days`: How far back to analyze conversations (default: 30)
- `min_mentions_for_interest`: Minimum mentions to track interest (default: 2)
- `confidence_threshold`: Minimum confidence to act on interests (default: 0.6)
- `max_interests_per_analysis`: Maximum interests to track (default: 10)

### Research Settings
- `auto_research_enabled`: Enable autonomous research cycles
- `research_cooldown_hours`: Cooldown between research cycles (default: 24)
- Rate limiting and monitoring configurations

## Testing and Development

### Test Files
- `dr3am_test_client.py`: Basic client testing
- `dr3am_testing_setup.py`: Setup for testing environments
- `integration_examples.py`: Integration examples and patterns

### Real API Integration
- `dr3am_with_real_apis.py`: Real API integration examples
- `real_api_integrations.py`: API integration patterns
- `dr3am_refl3ct_integration.py`: Refl3ct framework integration

## Safety Considerations

- Local-first processing - all analysis happens on your server
- No external data transmission unless explicitly configured
- Configurable boundaries to prevent unwanted research
- User consent mechanisms for research topics
- Audit trails for all autonomous actions

## Project Status & Completion Tasks

### What's Implemented ✅
- **Core MCP Server**: Full FastAPI implementation with MCP protocol compliance
- **Semantic Interest Analyzer**: LLM-powered conversation analysis with interest classification
- **Research Execution Engine**: Sophisticated research orchestration with real API integrations
- **Advanced Features**: ML predictions, multi-user support, framework integrations
- **Real API Integration**: Production-ready connections to Google, Bing, Amazon, eBay, etc.

### Critical Missing Components ❌
1. **Testing Infrastructure**
   - No pytest test suite
   - No integration or API testing
   - No load testing setup

2. **Production Requirements**
   - Missing `requirements.txt` with dependencies
   - No persistent database (uses in-memory storage)
   - No authentication/authorization system
   - Limited monitoring and logging

3. **Deployment Infrastructure**
   - No Docker/Kubernetes deployment configs
   - No CI/CD pipeline
   - Missing production deployment documentation

4. **Documentation Gaps**
   - No OpenAPI/Swagger API documentation
   - Missing developer setup guide
   - No troubleshooting documentation

### Immediate Actions Needed

#### High Priority
```bash
# Create requirements.txt
pip freeze > requirements.txt

# Add comprehensive test suite
mkdir tests/
# Create test_semantic_analyzer.py, test_mcp_server.py, etc.

# Set up persistent storage
# Convert in-memory storage to SQLite/PostgreSQL

# Add API documentation
# Generate OpenAPI docs from FastAPI
```

#### Medium Priority
- Consolidate mock vs real API implementations
- Add configuration validation and error handling
- Implement authentication system
- Create deployment scripts and documentation

#### Technical Debt
- Resolve potential circular import issues between modules
- Standardize error handling patterns across all components
- Add comprehensive type hints
- Implement graceful shutdown procedures

### Dependencies to Install
```bash
pip install fastapi uvicorn aiohttp pydantic numpy sqlite3
```

The project is feature-complete but needs productionization work for real-world deployment.

## Documentation

Comprehensive documentation is available in the `documentation/` directory:
- `dr3am_readme.md`: Complete project documentation
- `api_setup_guide.md`: API setup and configuration
- `complete_dr3am_deployment.md`: Deployment guide
- `refl3ct_dr3am_setup_guide.md`: Refl3ct integration setup