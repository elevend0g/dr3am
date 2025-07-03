# dr3am - Autonomous Agent MCP Server

Transform reactive chatbots into proactive research assistants through semantic interest analysis and autonomous research execution.

[![Coverage Status](https://codecov.io/gh/dr3am/dr3am/branch/main/graph/badge.svg)](https://codecov.io/gh/dr3am/dr3am)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is dr3am?

**dr3am** is an autonomous agent MCP (Model Context Protocol) server that transforms reactive chatbots into proactive research assistants. It analyzes conversation patterns to identify user interests and autonomously generates research opportunities when agents are idle.

### Key Features

- **🧠 Semantic Interest Analysis** - LLM-powered conversation analysis with interest classification
- **🔍 Autonomous Research** - Sophisticated research orchestration with real API integrations  
- **🔌 MCP Protocol Compliance** - Universal agent compatibility through standard protocol
- **🌐 Real API Integration** - Production-ready connections to Google, Bing, Amazon, eBay, etc.
- **👥 Multi-user Support** - Secure user management with JWT authentication
- **📊 Performance Monitoring** - Comprehensive metrics, logging, and health checks
- **🐳 Production Ready** - Docker deployment with monitoring stack

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/dr3am/dr3am.git
cd dr3am

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings

# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f dr3am
```

The application will be available at:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Database Admin**: http://localhost:5050 (pgAdmin)
- **Redis Admin**: http://localhost:8081

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-dev.txt

# Set up environment
cp .env.example .env
# Edit .env with your database and API keys

# Initialize database
python -c "from dr3am.models.database import init_database; init_database('sqlite:///dr3am.db')"

# Start development server
uvicorn dr3am.main:app --reload --host 0.0.0.0 --port 8000
```

## 📋 Requirements

- **Python**: 3.11+
- **Database**: PostgreSQL 15+ (SQLite for development)
- **Cache**: Redis 7+
- **Memory**: 512MB minimum, 2GB recommended
- **Storage**: 1GB minimum

## 🔧 Configuration

dr3am uses environment variables for configuration. Key settings include:

```bash
# Application
ENVIRONMENT=development
DEBUG=false
SECRET_KEY=your-secret-key-32-characters-minimum

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dr3am
REDIS_URL=redis://localhost:6379/0

# API Keys (optional but recommended)
GOOGLE_API_KEY=your-google-api-key
NEWS_API_KEY=your-news-api-key
LLM_API_KEY=your-llm-api-key

# Research Settings
DAILY_API_BUDGET=10.0
ANALYSIS_WINDOW_DAYS=30
CONFIDENCE_THRESHOLD=0.6
```

See `.env.example` for complete configuration options.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dr3am --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m api          # API tests only

# Run tests in Docker
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 📚 API Documentation

### Core Endpoints

- **POST** `/api/auth/login` - User authentication
- **GET** `/api/auth/me` - Current user information
- **POST** `/api/analyze` - Analyze conversations for interests
- **POST** `/api/research/trigger` - Trigger autonomous research
- **GET** `/api/interests/summary` - Get interest summary

### MCP Protocol Endpoints

- **POST** `/mcp/analyze_conversations` - Conversation analysis
- **POST** `/mcp/trigger_boredom` - Boredom-triggered research
- **POST** `/mcp/generate_research_plan` - Research plan generation

### Monitoring Endpoints

- **GET** `/health` - Health check
- **GET** `/metrics` - Prometheus metrics
- **GET** `/docs` - Interactive API documentation

## 🏗️ Architecture

```
dr3am/
├── dr3am/                          # Main package
│   ├── core/                       # Core components
│   │   ├── semantic_analyzer.py    # Interest analysis
│   │   ├── research_engine.py      # Research execution
│   │   └── mcp_server.py          # MCP protocol server
│   ├── api/                        # External API integrations
│   ├── auth/                       # Authentication system
│   ├── models/                     # Data models & database
│   ├── monitoring/                 # Metrics & health checks
│   └── utils/                      # Configuration & logging
├── tests/                          # Test suite
├── docs/                           # Documentation
└── scripts/                        # Deployment scripts
```

### Key Components

1. **Semantic Analyzer**: Analyzes conversations using LLMs to detect user interests
2. **Research Engine**: Orchestrates autonomous research based on detected interests  
3. **MCP Server**: Provides standard protocol interface for agent integration
4. **API Integrations**: Real connections to external services (Google, News, etc.)
5. **Authentication**: JWT-based user management and API security

## 🔌 Integration Examples

### LangChain

```python
from langchain.tools import tool
from dr3am import Dr3amClient

@tool
async def autonomous_research():
    """Trigger autonomous research based on conversation history"""
    async with Dr3amClient() as dr3am:
        return await dr3am.trigger_boredom()
```

### AutoGen

```python
import autogen

# Add dr3am as a tool in agent configuration
config_list = [{
    "model": "gpt-4",
    "tools": [
        {
            "type": "function", 
            "function": dr3am_research_tool
        }
    ]
}]
```

### Custom Integration

```python
from dr3am import Dr3amClient

async def integrate_with_agent():
    async with Dr3amClient() as dr3am:
        # Analyze recent conversations
        interests = await dr3am.analyze_conversations(conversation_history)
        
        # Generate research opportunities  
        opportunities = await dr3am.generate_research_plan(interests)
        
        # Execute autonomous research
        results = await dr3am.execute_research(opportunities[0])
        
        return results
```

## 🚀 Deployment

### Production Deployment

```bash
# Build production image
docker build --target production -t dr3am:prod .

# Deploy with production compose
docker-compose -f docker-compose.prod.yml up -d

# Monitor with Grafana (included)
open http://localhost:3000
```

### Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=dr3am

# View logs
kubectl logs -f deployment/dr3am
```

### Environment Variables for Production

```bash
ENVIRONMENT=production
SECRET_KEY=secure-32-character-secret-key
DATABASE_URL=postgresql://user:pass@prod-db:5432/dr3am
REDIS_URL=redis://prod-redis:6379/0
DAILY_API_BUDGET=100.0
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

## 📊 Monitoring

dr3am includes comprehensive monitoring:

- **Metrics**: Prometheus metrics for requests, performance, and business KPIs
- **Logging**: Structured JSON logging with request tracing
- **Health Checks**: Database, Redis, and API service health monitoring  
- **Dashboards**: Pre-configured Grafana dashboards
- **Alerting**: Configurable alerts for critical metrics

### Key Metrics

- Request rate and response times
- API usage and cost tracking
- Interest detection accuracy
- Research completion rates
- System resource usage

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/dr3am.git
cd dr3am

# Set up development environment  
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Start development server
uvicorn dr3am.main:app --reload
```

### Code Quality

- **Formatting**: Black, isort
- **Linting**: flake8, mypy  
- **Testing**: pytest with 90%+ coverage
- **Security**: bandit, safety
- **Pre-commit**: Automated checks on commit

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/dr3am/dr3am/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dr3am/dr3am/discussions)
- **Security**: security@dr3am.ai

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- SQLAlchemy for robust ORM capabilities  
- Pydantic for data validation
- The open source community for inspiration and contributions

---

**Ready to build the future of autonomous agents!** 🚀