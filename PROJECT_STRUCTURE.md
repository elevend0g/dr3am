# dr3am Project Structure

This document describes the clean, production-ready structure of the dr3am project.

## 📁 Root Directory

```
dr3am/
├── 📋 Project Files
│   ├── README.md                    # Main project documentation
│   ├── CONTRIBUTING.md              # Contribution guidelines
│   ├── LICENSE                      # MIT license
│   ├── CLAUDE.md                    # Claude Code instructions
│   └── PROJECT_STRUCTURE.md         # This file
│
├── 🐍 Python Package
│   ├── setup.py                     # Package configuration
│   ├── requirements.txt             # Production dependencies
│   ├── requirements-dev.txt         # Development dependencies
│   └── pytest.ini                   # Test configuration
│
├── 🐳 Docker & Deployment
│   ├── Dockerfile                   # Multi-stage container
│   ├── docker-compose.yml           # Development environment
│   ├── docker-compose.prod.yml      # Production environment
│   ├── docker-compose.test.yml      # Testing environment
│   └── .dockerignore                # Docker ignore patterns
│
├── 🔧 Configuration
│   ├── .env                         # Development environment vars
│   ├── .env.example                 # Environment template
│   ├── .gitignore                   # Git ignore patterns
│   └── .pre-commit-config.yaml      # Code quality hooks
│
├── 🏗️ CI/CD
│   └── .github/
│       └── workflows/
│           ├── ci.yml               # Continuous integration
│           └── release.yml          # Release automation
│
├── 📦 Source Code
│   └── dr3am/                       # Main package
│
├── 🧪 Testing
│   └── tests/                       # Test suite
│
├── 📜 Scripts
│   └── scripts/                     # Deployment scripts
│
└── 📁 Archive
    └── archive/                     # Legacy files (can be deleted)
```

## 📦 Source Code Structure (`dr3am/`)

```
dr3am/
├── __init__.py                      # Package initialization
├── main.py                          # FastAPI application entry point
│
├── 🧠 core/                         # Core business logic
│   ├── __init__.py
│   ├── semantic_analyzer.py         # Interest analysis engine
│   ├── research_engine.py           # Research orchestration
│   └── mcp_server.py                # MCP protocol server
│
├── 🔌 api/                          # External API integrations
│   ├── __init__.py
│   └── integrations.py              # Real API connections
│
├── 🔐 auth/                         # Authentication system
│   ├── __init__.py
│   ├── models.py                    # Auth data models
│   ├── routes.py                    # Auth endpoints
│   └── security.py                 # JWT & security utilities
│
├── 📊 models/                       # Data models & database
│   ├── __init__.py
│   ├── database.py                  # SQLAlchemy models
│   ├── conversation.py              # Conversation models
│   ├── interests.py                 # Interest models
│   ├── research.py                  # Research models
│   ├── alembic.ini                  # Database migration config
│   └── migrations/                  # Database migrations
│       ├── env.py
│       └── script.py.mako
│
├── 📈 monitoring/                   # Observability
│   ├── __init__.py
│   └── middleware.py                # Request logging & metrics
│
└── 🛠️ utils/                        # Utilities
    ├── __init__.py
    ├── config.py                    # Configuration management
    └── logging.py                   # Structured logging
```

## 🧪 Test Structure (`tests/`)

```
tests/
├── __init__.py
├── conftest.py                      # Test configuration & fixtures
├── fixtures/                       # Test data fixtures
├── unit/                           # Unit tests
│   ├── test_models.py
│   ├── test_semantic_analyzer.py
│   ├── test_research_engine.py
│   └── test_mcp_server.py
├── integration/                    # Integration tests
└── api/                           # API endpoint tests
```

## 📜 Scripts (`scripts/`)

```
scripts/
├── docker-build.sh                 # Docker build automation
└── init-db.sql                     # Database initialization
```

## 📁 Archive Directory

The `archive/` directory contains legacy files from the original development:

- `archive/code/` - Original prototype implementations
- `archive/documentation/` - Legacy documentation files

**Note**: These files can be safely deleted once you're confident the migration is complete.

## 🔄 Migration Summary

### What Was Moved to Archive:
- ✅ `code/` directory - All original Python files
- ✅ `documentation/` directory - Legacy documentation

### What Was Cleaned Up:
- ✅ Empty directories removed
- ✅ Duplicate files consolidated
- ✅ Proper package structure implemented
- ✅ Production-ready configuration added

### What's Production-Ready:
- ✅ Structured Python package with proper imports
- ✅ Comprehensive test suite with fixtures
- ✅ Docker multi-stage builds
- ✅ Database models with migrations
- ✅ Authentication and security
- ✅ Structured logging and monitoring
- ✅ CI/CD pipelines
- ✅ Complete documentation

## 🚀 Quick Start

```bash
# Clone and set up
git clone <repository>
cd dr3am
cp .env.example .env

# Development with Docker
docker-compose up -d

# Or local development
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
uvicorn dr3am.main:app --reload
```

## 📚 Key Resources

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Project Documentation**: [README.md](README.md)
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **License**: [LICENSE](LICENSE)

The project is now clean, well-organized, and ready for production deployment! 🎉