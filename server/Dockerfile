# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU version first (as per requirements.txt comment)
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY search.py .
COPY .env .
COPY static/ ./static/
COPY templates/ ./templates/

# Expose port
EXPOSE 5000

# Make search.py executable
RUN chmod +x search.py

# Run the application
CMD ["python", "search.py"]
