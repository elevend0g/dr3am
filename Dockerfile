# Multi-stage Dockerfile for dr3am
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user and directories
RUN useradd --create-home --shell /bin/bash dr3am && \
    mkdir -p /app /app/logs && \
    chown -R dr3am:dr3am /app

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY --chown=dr3am:dr3am . .

USER dr3am

# Expose ports
EXPOSE 8000 9090

# Default command for development
CMD ["uvicorn", "dr3am.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=dr3am:dr3am dr3am/ ./dr3am/
COPY --chown=dr3am:dr3am setup.py ./
COPY --chown=dr3am:dr3am README.md ./
COPY --chown=dr3am:dr3am .env.example ./

# Install the package
RUN pip install -e .

# Switch to app user
USER dr3am

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Production command
CMD ["gunicorn", "dr3am.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Testing stage
FROM development as testing

# Copy test files
COPY --chown=dr3am:dr3am tests/ ./tests/
COPY --chown=dr3am:dr3am pytest.ini ./

# Run tests
RUN python -m pytest tests/ --cov=dr3am --cov-report=xml --cov-report=term-missing

# Default stage
FROM production