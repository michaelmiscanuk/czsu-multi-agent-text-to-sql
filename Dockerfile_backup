# ==============================================================================
# Production-Ready Dockerfile for CZSU Multi-Agent Text-to-SQL API
# Uses Debian Slim for better compatibility with ChromaDB and ML/AI packages
# ==============================================================================

# Stage 1: Build stage with Debian Slim for full ChromaDB compatibility
FROM python:3.11-slim-bookworm AS builder

# Set build arguments for optimization
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Install essential build dependencies for Debian + ML/AI packages
# Required for chromadb, psutil, and other ML dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    libpq-dev \
    libsqlite3-dev \
    build-essential \
    pkg-config \
    curl \
    git \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements.txt for dependency management
COPY requirements.txt ./

# Install dependencies with Debian optimized strategy
# The full chromadb package works much better on Debian than Alpine
RUN pip install --no-cache-dir -r requirements.txt

# Clean up build dependencies to reduce image size
RUN apt-get remove --purge -y gcc g++ build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip cache purge

# ==============================================================================
# Stage 2: Production runtime with Debian Slim (matching build stage)
# ==============================================================================
FROM python:3.11-slim-bookworm AS production

# Install only essential runtime dependencies for Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libsqlite3-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (Debian)
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables for optimal performance and security
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONHASHSEED=random \
    PYTHONMALLOC=malloc \
    PYTHONGC=1 \
    # Memory optimization for glibc malloc
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TRIM_THRESHOLD_=131072 \
    MALLOC_TOP_PAD_=131072 \
    MALLOC_MMAP_MAX_=65536 \
    # Uvicorn/FastAPI optimization for Railway
    WEB_CONCURRENCY=1 \
    MAX_WORKERS=1 \
    WORKER_CLASS=uvicorn.workers.UvicornWorker \
    KEEP_ALIVE=2 \
    # Application-specific
    PORT=8000 \
    HOST=0.0.0.0

# Set working directory and switch to non-root user
WORKDIR /app
USER appuser

# Copy application code with proper ownership
COPY --chown=appuser:appgroup . .

# Run unzip process and cleanup zip files (matching your build command)
RUN python unzip_files.py && rm -f data/*.zip metadata/*.zip

# Health check optimized for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health', timeout=5)" || exit 1

# Expose port (Railway will override with $PORT)
EXPOSE $PORT

# Optimized startup command for Railway
# Uses Python uvicorn.run() like local development but optimized for production
CMD python -c "\
import os; \
import uvicorn; \
uvicorn.run( \
    'api.main:app', \
    host=os.getenv('HOST', '0.0.0.0'), \
    port=int(os.getenv('PORT', 8000)), \
    workers=1, \
    log_level='info', \
    access_log=False, \
    use_colors=False, \
    reload=False \
)"