# Docker Deployment Guide - Adobe Hackathon Round 1A

## ✅ Docker Requirements Compliance

### Architecture & Platform

- ✅ **AMD64 Architecture**: Dockerfile explicitly specifies `--platform=linux/amd64`
- ✅ **CPU Only**: No GPU dependencies, optimized for CPU execution
- ✅ **Resource Compatible**: Tested on 8 CPU, 16GB RAM configurations

### Performance & Constraints

- ✅ **Execution Time**: <10 seconds per 50-page PDF (currently ~0.11s per file)
- ✅ **Model Size**: All dependencies < 200MB total (no large ML models)
- ✅ **Offline Operation**: No internet calls, works with `--network none`

### Expected Execution Commands

#### Build Command

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

#### Run Command

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier
```

**Note for Windows Users (Git Bash/MINGW64):**

When running the above bash command in Git Bash or MINGW64 on Windows, Docker for Windows may misinterpret the `$(pwd)` path (which is Unix-style) and create folders like `input;C` and `output;C` instead of using your actual `input` and `output` directories. This is due to a known path translation issue between Git Bash and Docker Desktop on Windows.

**Workaround:**

- Use `$(pwd -W)` instead of `$(pwd)` in the command to provide the correct Windows-style path:

  ```bash
  docker run --rm \
    -v "$(pwd -W)/input:/app/input" \
    -v "$(pwd -W)/output:/app/output" \
    --network none \
    mysolutionname:somerandomidentifier
  ```

- Alternatively, run the command in PowerShell using `${PWD}` for correct path mapping:

  ```powershell
  docker run --rm -v "${PWD}\input:/app/input" -v "${PWD}\output:/app/output" --network none mysolutionname:somerandomidentifier
  ```

If the evaluation team uses the original bash command on Windows, this may result in incorrect folder mapping and missing outputs. On Linux/macOS, the original command works as intended.

## 🐳 Container Behavior

### Automatic Processing

- **Input**: Container automatically scans `/app/input` for PDF files
- **Processing**: Converts each `filename.pdf` to `filename.json`
- **Output**: Generated JSON files saved to `/app/output`
- **No Manual Intervention**: Fully automated batch processing

### Volume Mounting

- **Input Volume**: `-v $(pwd)/input:/app/input` (read PDFs from host)
- **Output Volume**: `-v $(pwd)/output:/app/output` (write JSONs to host)
- **Network Isolation**: `--network none` (offline operation)

## 📋 Implementation Details

### Dockerfile Features

```dockerfile
FROM --platform=linux/amd64 python:3.10-slim
# ... dependencies installation ...
CMD ["python", "main.py", "/app/input", "/app/output", "--batch"]
```

### Main Script Integration

- Auto-detects Docker environment
- Default batch processing mode for `/app/input` → `/app/output`
- Comprehensive error handling and logging
- Performance monitoring and statistics

### Dependencies

- **PyMuPDF**: Primary PDF parser (fast, reliable)
- **PDFMiner**: Fallback parser for complex documents
- **Tesseract**: OCR support for image-based PDFs
- **Lightweight packages**: All requirements optimized for size

## 🧪 Testing & Validation

### Pre-deployment Validation

```bash
python validate_docker.py
```

**Result**: ✅ 5/5 checks passed

### Batch Processing Test

```bash
python main.py input output --batch
```

**Result**: ✅ 5 PDFs processed in 0.55s (avg 0.11s per file)

## 📊 Performance Metrics

- **Processing Speed**: 0.11s average per PDF file
- **Success Rate**: 100% (5/5 files processed successfully)
- **Memory Usage**: Optimized for container constraints
- **Parser Usage**: PyMuPDF primary (100% success rate)

## 🚀 Ready for Deployment

The solution is fully Docker-ready and meets all hackathon requirements:

1. ✅ **AMD64 compatible** with explicit platform specification
2. ✅ **Offline operation** with no internet dependencies
3. ✅ **Performance compliant** (<10s per 50-page PDF)
4. ✅ **Resource efficient** (<200MB total dependencies)
5. ✅ **Automated processing** of input directory to output directory
6. ✅ **Proper JSON output** format matching schema requirements

**Next Step**: Deploy using the provided Docker commands for evaluation.
