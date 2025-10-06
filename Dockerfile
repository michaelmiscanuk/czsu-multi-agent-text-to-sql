# ==============================================================================
# Production-Ready Dockerfile for CZSU Multi-Agent Text-to-SQL API
# Uses Alpine Linux with proper musl/glibc compatibility for ML/AI packages
# ==============================================================================

# Stage 1: Build stage with Alpine for minimal size + full build tools
FROM python:3.11-alpine3.19 AS builder

# Set build arguments for optimization
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Install comprehensive build dependencies for Alpine + ML/AI packages
# Key packages for psutil: gcc, musl-dev, linux-headers, python3-dev
# Key packages for chromadb/onnxruntime: build-base, cmake, git, libffi-dev
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    python3-dev \
    build-base \
    libffi-dev \
    openssl-dev \
    postgresql-dev \
    sqlite-dev \
    curl \
    pkgconfig \
    cmake \
    git \
    make \
    autoconf \
    automake \
    libtool \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Rust via rustup (for cryptography and other Rust-based packages)
# Required for many Python packages that have Rust components
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . ~/.cargo/env \
    && rustup default stable

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set environment variables for Alpine/musl compilation
ENV CFLAGS="-Os -w" \
    CXXFLAGS="-Os -w" \
    LDFLAGS="-Wl,--strip-all" \
    CC="gcc" \
    CXX="g++" \
    PYTHONHASHSEED=random \
    MAKEFLAGS="-j$(nproc)"

# Copy requirements.txt for dependency management
COPY requirements.txt ./

# Install dependencies with Alpine-optimized strategy
# 1. Install psutil first from source (Alpine compatibility)  
# 2. Install onnxruntime from Alpine's native package (musl compatible)
# 3. Install all other dependencies from requirements.txt

# Install psutil specifically with source compilation for Alpine
RUN pip install --no-cache-dir \
    --no-binary psutil \
    --compile \
    psutil>=5.9.0

# Install onnxruntime from Alpine's native package (musl libc compatible)
# Enable testing repository and install the native Alpine package
RUN echo "https://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories \
    && apk add --no-cache py3-onnxruntime \
    && python -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')"

# Install all other dependencies
RUN pip install --no-cache-dir \
    --prefer-binary \
    -r requirements.txt

# Clean up build dependencies and caches to reduce image size
RUN apk del .build-deps \
    && rm -rf /root/.cargo \
    && rm -rf /tmp/* \
    && rm -rf /var/cache/apk/* \
    && pip cache purge

# ==============================================================================
# Stage 2: Production runtime with Alpine (matching build stage)
# ==============================================================================
FROM python:3.11-alpine3.19 AS production

# Install only essential runtime dependencies for Alpine
RUN apk add --no-cache \
    libpq \
    libffi \
    openssl \
    sqlite \
    libgcc \
    libstdc++ \
    musl \
    && rm -rf /var/cache/apk/* \
    && rm -rf /tmp/*

# Create non-root user for security (Alpine)
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