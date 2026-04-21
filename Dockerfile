FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY data/raw/ ./data/raw/

# Create data directories
RUN mkdir -p data/raw data/processed models reports/figures

# Expose port
EXPOSE 8000

# Default command: run API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
