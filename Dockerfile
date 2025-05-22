# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, including git for pip git installs
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files for the API
COPY app/ /app/app/
COPY models/ /app/models/
COPY src/ /app/src/
COPY conf/ /app/conf/

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
