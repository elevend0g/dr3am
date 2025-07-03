#!/bin/bash
# Docker build script for dr3am

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
TARGET="development"
TAG="latest"
PUSH=false
PLATFORM=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -t, --target TARGET    Build target (development, production, testing)"
            echo "  --tag TAG             Docker tag (default: latest)"
            echo "  --push                Push image to registry"
            echo "  --platform PLATFORM   Target platform (e.g., linux/amd64,linux/arm64)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate target
if [[ "$TARGET" != "development" && "$TARGET" != "production" && "$TARGET" != "testing" ]]; then
    print_error "Invalid target: $TARGET. Must be one of: development, production, testing"
    exit 1
fi

print_status "Building dr3am Docker image..."
print_status "Target: $TARGET"
print_status "Tag: dr3am:$TAG"

# Build command
BUILD_CMD="docker build $PLATFORM --target $TARGET -t dr3am:$TAG ."

if [[ "$TARGET" == "production" ]]; then
    BUILD_CMD="$BUILD_CMD -t dr3am:$TAG-prod"
fi

print_status "Running: $BUILD_CMD"

# Execute build
if eval $BUILD_CMD; then
    print_status "Build completed successfully!"
else
    print_error "Build failed!"
    exit 1
fi

# Push if requested
if [[ "$PUSH" == true ]]; then
    print_status "Pushing image to registry..."
    
    if docker push dr3am:$TAG; then
        print_status "Push completed successfully!"
    else
        print_error "Push failed!"
        exit 1
    fi
    
    if [[ "$TARGET" == "production" ]]; then
        if docker push dr3am:$TAG-prod; then
            print_status "Production tag push completed successfully!"
        else
            print_error "Production tag push failed!"
            exit 1
        fi
    fi
fi

print_status "Docker build script completed!"

# Show image info
print_status "Image information:"
docker images dr3am:$TAG