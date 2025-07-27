@echo off
REM Docker test script for Adobe Hackathon Round 1A (Windows)

echo === Docker Build Test ===
echo Building Docker image with AMD64 platform...

REM Build the Docker image
docker build --platform linux/amd64 -t adobe-hackathon-round1a:test .

if %errorlevel% equ 0 (
    echo ✅ Docker image built successfully
) else (
    echo ❌ Docker build failed
    exit /b 1
)

echo.
echo === Docker Run Test ===
echo Testing Docker execution with mounted volumes...

REM Create test directories if they don't exist
if not exist test_input mkdir test_input
if not exist test_output mkdir test_output

REM Copy test PDFs to test_input
if exist input\*.pdf (
    copy input\*.pdf test_input\ >nul 2>&1
) else (
    echo No PDF files found in input directory
)

REM Run the Docker container
docker run --rm -v "%cd%/test_input:/app/input" -v "%cd%/test_output:/app/output" --network none adobe-hackathon-round1a:test

if %errorlevel% equ 0 (
    echo ✅ Docker container executed successfully
    echo.
    echo Output files generated:
    dir test_output
) else (
    echo ❌ Docker execution failed
    exit /b 1
)

echo.
echo === Cleanup ===
REM Clean up test directories (optional)
REM rmdir /s /q test_input test_output

echo Docker test completed successfully! 🎉
