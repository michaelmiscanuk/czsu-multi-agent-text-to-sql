# 🐳 Docker Deployment Guide

## Overview

This guide covers the production-ready Alpine Docker setup for the CZSU Multi-Agent Text-to-SQL API, optimized for Railway deployment with minimal memory footprint.

## 🏗️ Docker Architecture

### Multi-Stage Build
- **Builder Stage**: Compiles all heavy ML/AI dependencies (LangChain, ChromaDB, etc.)
- **Production Stage**: Minimal runtime with only essential libraries
- **Result**: ~70% smaller final image compared to Ubuntu-based builds

### Memory Optimizations
- Alpine Linux base (~5MB vs Ubuntu ~72MB)
- Virtual environment isolation
- Optimized Python memory settings
- Single worker configuration for Railway
- Garbage collection tuning

## 📁 Key Files

```
├── Dockerfile              # Multi-stage Alpine build
├── .dockerignore           # Excludes unnecessary files
├── docker-build-test.sh    # Build/test script (Linux/Mac)
├── docker-build-test.bat   # Build/test script (Windows)
└── railway.toml            # Railway deployment config
```

## 🚀 Local Testing

### Windows:
```cmd
# Build and test
docker-build-test.bat full

# Individual commands
docker-build-test.bat build
docker-build-test.bat run
docker-build-test.bat test
docker-build-test.bat clean
```

### Linux/Mac:
```bash
# Make executable
chmod +x docker-build-test.sh

# Build and test
./docker-build-test.sh full

# Individual commands
./docker-build-test.sh build
./docker-build-test.sh run
./docker-build-test.sh test
./docker-build-test.sh clean
```

## 🚂 Railway Deployment

### Prerequisites
1. Push Dockerfile to your repository
2. Railway account with project connected to your repo

### Deployment Steps
1. **Enable Docker**: In Railway dashboard → Settings → Enable "Use Dockerfile"
2. **Environment Variables**: Add required secrets:
   ```
   OPENAI_API_KEY=your_key
   DATABASE_URL=your_db_url
   LANGSMITH_API_KEY=your_key
   # ... other environment variables
   ```
3. **Deploy**: Railway will automatically use your Dockerfile

### Railway Configuration
The `railway.toml` file includes:
- Memory-optimized resource limits
- Health check configuration
- Production environment variables

## 🔧 Technical Details

### Build Dependencies (Alpine packages)
- `gcc`, `g++`, `musl-dev` - C/C++ compilation
- `postgresql-dev` - PostgreSQL client libraries
- `rust`, `cargo` - For cryptography packages
- `cmake`, `openblas-dev` - For NumPy/scientific packages

### Runtime Dependencies (minimal)
- `libpq` - PostgreSQL client library
- `libffi`, `openssl` - Cryptography support
- `sqlite` - SQLite database support
- `openblas`, `lapack` - Math libraries for ML

### Memory Environment Variables
```dockerfile
MALLOC_ARENA_MAX=2              # Limit memory arenas
PYTHONMALLOC=malloc             # Use system malloc
WEB_CONCURRENCY=1               # Single worker process
PYTHONDONTWRITEBYTECODE=1       # No .pyc files
PYTHONGC=1                      # Enable garbage collection
```

## 📊 Expected Performance

### Memory Usage
- **Base OS**: ~70% reduction (Alpine vs Ubuntu)
- **Python Runtime**: ~20-30% reduction from optimizations
- **Build Artifacts**: Eliminated (multi-stage build)
- **Overall**: **40-60% memory reduction** expected

### Build Time
- **Initial Build**: 5-10 minutes (compiling heavy ML packages)
- **Subsequent Builds**: 2-3 minutes (Docker layer caching)
- **Railway Deploy**: 3-5 minutes (including push and deploy)

## ⚠️ Important Notes

### Data Persistence
The Dockerfile creates empty placeholder files for:
- `data/czsu_data.db` - SQLite database
- `metadata/selection_descriptions.csv` - Selection metadata
- `metadata/czsu_chromadb/` - Vector database directory

**For production**: Mount persistent volumes or use external databases.

### Security
- ✅ Non-root user (`appuser:appgroup`)
- ✅ Minimal attack surface (Alpine + essential packages only)
- ✅ No unnecessary build tools in final image
- ✅ Read-only filesystem compatible

### Windows Event Loop
The application code handles Windows-specific async event loop, but Docker uses Linux, so this is handled gracefully in the code.

## 🐛 Troubleshooting

### Build Issues
```bash
# Clear Docker cache
docker system prune -a

# Build with verbose output
docker build -t czsu-multi-agent-api . --progress=plain --no-cache
```

### Runtime Issues
```bash
# Check container logs
docker logs czsu-api-test

# Interactive shell access
docker exec -it czsu-api-test /bin/sh

# Memory usage monitoring
docker stats czsu-api-test
```

### Railway Issues
1. **Build Timeout**: Railway has build time limits; heavy ML packages may need optimization
2. **Memory Limits**: Adjust memory settings in Railway dashboard if needed
3. **Port Binding**: Railway automatically sets `$PORT`; Dockerfile handles this

## 📈 Monitoring

The health check endpoint `/health` runs every 30 seconds and includes:
- Application startup verification
- Basic dependency checks
- Memory usage monitoring

Monitor in Railway dashboard or via:
```bash
curl http://your-app.railway.app/health
```

## 🔄 Updates

To update the deployment:
1. Push changes to your repository
2. Railway automatically rebuilds and deploys
3. Zero-downtime deployment with health checks

The Docker setup ensures consistent, reproducible deployments across all environments.