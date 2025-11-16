FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data /app/logs

# Run as non-root user
RUN useradd -m -u 1000 brain && chown -R brain:brain /app
USER brain

CMD ["python", "main.py"]