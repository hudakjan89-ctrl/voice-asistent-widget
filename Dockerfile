# Force rebuild v3 - add curl + fix healthcheck
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies including curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python files
COPY main.py .
COPY config.py .
COPY errors.py .
COPY text_normalizer.py .

# Create client directory
RUN mkdir -p /app/client

# Copy client folder contents explicitly
COPY ./client /app/client/

# Verify client folder and index.html - FAIL BUILD if not present
RUN echo "=== Verifying client files ===" && \
    ls -la /app/ && \
    echo "=== Client directory ===" && \
    ls -la /app/client/ && \
    echo "=== Checking index.html exists ===" && \
    test -f /app/client/index.html || (echo "ERROR: index.html NOT FOUND!" && exit 1) && \
    echo "=== index.html content preview ===" && \
    head -5 /app/client/index.html && \
    echo "=== Verification PASSED ==="

# Verify module can be imported
RUN python -c "import main; print('Module import OK')"

# Expose port
EXPOSE 8080

# Health check - FIXED PORT 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "debug"]
