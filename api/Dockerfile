# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /workspace/api

# Install system dependencies for Pillow and other packages
# RUN apt-get update && apt-get install -y \
    # build-essential \
    # libpq-dev \
    # libjpeg-dev \
    # libpng-dev \
    # libtiff-dev \
    # libfreetype6-dev \
    # liblcms2-dev \
    # zlib1g-dev \
    # && apt-get clean \
    # && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Expose FastAPI port
EXPOSE 5002

# Start with uvicorn (FastAPI 공식 실행 방식)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5002"]

