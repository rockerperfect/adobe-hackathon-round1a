# Adobe Hackathon Round 1A: PDF Outline Extraction System

## Approach

This solution is designed for robust, fully automated PDF outline extraction, meeting all hackathon requirements:

- **Generalized, non-hardcoded logic** for heading/title detection and outline building
- **Batch processing**: Processes all PDFs in the input directory and outputs JSONs to the output directory
- **Event/flyer/form detection**: Filters out non-standard documents (e.g., event flyers, forms) to avoid false headings
- **OCR integration**: Automatically detects and processes scanned/image-based PDFs using Tesseract OCR
- **Adaptive thresholds**: Uses statistical and clustering techniques to adapt to each document's font and layout
- **Offline, Dockerized, AMD64-compatible**: Runs fully offline in a Docker container, with all dependencies included
- **No manual intervention**: Container auto-processes all files and exits

## Models and Libraries Used

### Core Libraries

- **PyMuPDF (`fitz`)**: Fast PDF parsing and image extraction
- **PDFMiner.six**: Fallback PDF parsing for complex/edge-case documents
- **pytesseract**: OCR for scanned/image-based PDFs (uses Tesseract engine)
- **OpenCV (`cv2`)**: Image preprocessing for OCR (binarization, denoising, etc.)
- **Pillow (`PIL`)**: Image manipulation and enhancement
- **scikit-learn**: Statistical analysis, clustering (for adaptive font/heading detection)
- **pandas, numpy**: Data handling and numerical operations

### Other Python Standard Libraries

- argparse, json, logging, os, sys, time, pathlib, typing, dataclasses, collections, re, math, statistics, enum

### No external ML models are used

- All logic is algorithmic/statistical; OCR is handled by Tesseract

## How to Build and Run

### Expected Execution (as per hackathon instructions)

```bash
# Build the Docker image (AMD64 compatible)
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

# Run the container with mounted volumes (offline mode)
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier
```

> **Note for Windows Users (Git Bash/MINGW64):**
> If you run the above bash command in Git Bash on Windows, Docker may misinterpret the path and create folders like `input;C` and `output;C`. This is a known Docker Desktop path translation issue. For correct results on Windows, use `$(pwd -W)` instead of `$(pwd)` in the command, or run the command in PowerShell as shown below:
>
> **Git Bash (Windows):**
>
> ```bash
> docker run --rm \
>   -v "$(pwd -W)/input:/app/input" \
>   -v "$(pwd -W)/output:/app/output" \
>   --network none \
>   mysolutionname:somerandomidentifier
> ```
>
> **PowerShell (Windows):**
>
> ```powershell
> docker run --rm -v "${PWD}\input:/app/input" -v "${PWD}\output:/app/output" --network none mysolutionname:somerandomidentifier
> ```
>
> On Linux/macOS, the original command works as intended.

### What Happens

- All PDFs in the `input/` directory are processed
- For each `filename.pdf`, a corresponding `filename.json` is created in `output/`
- The container runs fully offline and removes itself after completion

### Dependencies

All dependencies are installed automatically inside the Docker container via `requirements.txt`:

- PyMuPDF, pdfminer.six, pytesseract, opencv-python-headless, Pillow, scikit-learn, pandas, numpy
- Tesseract OCR engine is installed in the Dockerfile for OCR support

### No manual intervention is required

- Place your PDFs in the `input/` folder, run the above commands, and collect your JSONs from `output/`

---

## Additional Notes

- The solution is robust to a wide variety of PDF layouts, including scanned/image-based, multi-column, and multilingual documents
- All code is generalized—no hardcoded patterns or document-specific logic
- Event flyers and forms are detected and filtered to avoid false positives in the outline
- Performance: <10 seconds per 50-page PDF (tested on 8 CPU, 16GB RAM)
- Fully offline, resource-efficient, and Docker-ready

---

## Contact

For any issues or clarifications, please refer to the hackathon instructions or contact the organizers.
