FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies for multilingual support (AMD64 compatible)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-chi-sim \
    tesseract-ocr-ara \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create required directories
RUN mkdir -p /app/input /app/output

# Set entry point to automatically process /app/input to /app/output
CMD ["python", "main.py", "/app/input", "/app/output", "--batch"]
