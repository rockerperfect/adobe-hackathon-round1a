#!/bin/bash
# Docker test script for Adobe Hackathon Round 1A

echo "=== Docker Build Test ==="
echo "Building Docker image with AMD64 platform..."

# Build the Docker image
docker build --platform linux/amd64 -t adobe-hackathon-round1a:test .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

echo ""
echo "=== Docker Run Test ==="
echo "Testing Docker execution with mounted volumes..."

# Create test directories if they don't exist
mkdir -p test_input test_output

# Copy test PDFs to test_input
if [ -d "input" ]; then
    cp input/*.pdf test_input/ 2>/dev/null || echo "No PDF files found in input directory"
fi

# Run the Docker container
docker run --rm \
    -v $(pwd)/test_input:/app/input \
    -v $(pwd)/test_output:/app/output \
    --network none \
    adobe-hackathon-round1a:test

if [ $? -eq 0 ]; then
    echo "✅ Docker container executed successfully"
    echo ""
    echo "Output files generated:"
    ls -la test_output/
else
    echo "❌ Docker execution failed"
    exit 1
fi

echo ""
echo "=== Cleanup ==="
# Clean up test directories (optional)
# rm -rf test_input test_output

echo "Docker test completed successfully! 🎉"
