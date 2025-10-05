#!/bin/bash
# Docker build and test script for CZSU Multi-Agent Text-to-SQL API
# Usage: ./docker-build-test.sh [build|run|test|clean]

set -e

IMAGE_NAME="czsu-multi-agent-api"
CONTAINER_NAME="czsu-api-test"
PORT="8000"

case "${1:-build}" in
    "build")
        echo "🏗️  Building Docker image..."
        docker build -t $IMAGE_NAME . --no-cache
        echo "✅ Build completed successfully!"
        ;;
    
    "run")
        echo "🚀 Running container..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        docker run -d \
            --name $CONTAINER_NAME \
            -p $PORT:$PORT \
            -e PYTHONUNBUFFERED=1 \
            -e PORT=$PORT \
            $IMAGE_NAME
        echo "✅ Container started on http://localhost:$PORT"
        echo "📋 View logs: docker logs -f $CONTAINER_NAME"
        ;;
    
    "test")
        echo "🧪 Testing container health..."
        sleep 10  # Wait for startup
        if curl -f http://localhost:$PORT/health; then
            echo "✅ Health check passed!"
        else
            echo "❌ Health check failed!"
            docker logs $CONTAINER_NAME
            exit 1
        fi
        ;;
    
    "clean")
        echo "🧹 Cleaning up..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        docker rmi $IMAGE_NAME 2>/dev/null || true
        echo "✅ Cleanup completed!"
        ;;
    
    "full")
        echo "🔄 Full build, run, and test cycle..."
        $0 clean
        $0 build
        $0 run
        $0 test
        echo "✅ Full cycle completed successfully!"
        ;;
    
    *)
        echo "Usage: $0 [build|run|test|clean|full]"
        echo "  build - Build the Docker image"
        echo "  run   - Run the container"
        echo "  test  - Test container health"
        echo "  clean - Clean up containers and images"
        echo "  full  - Complete build, run, and test cycle"
        exit 1
        ;;
esac