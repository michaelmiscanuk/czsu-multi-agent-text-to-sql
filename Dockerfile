# ==============================================================================
# Production-Ready Dockerfile for CZSU Multi-Agent Text-to-SQL API
# Uses Debian for build (glibc compatibility) and Alpine for runtime (minimal size)
# ==============================================================================

# Stage 1: Build stage with Debian for glibc compatibility (onnxruntime, chromadb)
FROM python:3.11-slim-bookworm AS builder

# Set build arguments
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Install comprehensive build dependencies for heavy ML/AI packages (Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    libsqlite3-dev \
    curl \
    pkg-config \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel uv

# Install Rust via rustup (for cryptography and other Rust-based packages)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && source ~/.cargo/env \
    && rustup default stable

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy pyproject.toml and uv.lock for dependency management
COPY pyproject.toml uv.lock* ./

# Install dependencies with uv (faster and more reliable than pip)
RUN uv pip install \
    --system \
    --find-links https://download.pytorch.org/whl/cpu \
    .

# Clean up build dependencies to reduce layer size
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Production runtime with minimal Alpine
# ==============================================================================
FROM python:3.11-alpine3.19 AS production

# Install only essential runtime dependencies
RUN apk add --no-cache \
    libpq \
    libffi \
    openssl \
    sqlite \
    openblas \
    lapack \
    && rm -rf /var/cache/apk/* \
    && rm -rf /tmp/*

# Create non-root user for security
RUN addgroup -g 1000 -S appgroup && \
    adduser -u 1000 -S appuser -G appgroup -h /app -s /bin/sh

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

# Create necessary data directories that the app expects
RUN mkdir -p data metadata logs && \
    # Ensure required data files exist (create empty if missing)
    touch data/czsu_data.db && \
    touch metadata/selection_descriptions.csv && \
    # Create chromadb directory structure
    mkdir -p metadata/czsu_chromadb && \
    mkdir -p metadata/schemas

# Health check optimized for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health', timeout=5)" || exit 1

# Expose port (Railway will override with $PORT)
EXPOSE $PORT

# Optimized startup command for Railway
# Uses Railway's $PORT environment variable, single worker, reduced logging
CMD python -m uvicorn api.main:app \
    --host $HOST \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --no-access-log \
    --loop uvloop \
    --http httptools