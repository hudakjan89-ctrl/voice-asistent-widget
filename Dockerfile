FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files explicitly
COPY main.py .
COPY config.py .
COPY errors.py .
COPY text_normalizer.py .

# Copy client folder explicitly
COPY client/ ./client/

# Verify client folder contents
RUN echo "=== Verifying client folder ===" && \
    ls -la /app/client/ && \
    echo "=== client/index.html exists: ===" && \
    cat /app/client/index.html | head -20

# List all files for debugging
RUN echo "=== All files in /app ===" && ls -la /app

# Verify module can be imported
RUN python -c "import main; print('Module import OK')"

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the application with debug logging
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "debug"]
