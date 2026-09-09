# -- VittaBodh Dockerfile --------------------------------------------------
# Uses the official TensorFlow base image (CPU-only, ~1.5 GB).
# For GPU support, change to tensorflow/tensorflow:2.16.0-gpu.

FROM tensorflow/tensorflow:2.16.1

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy application code
COPY . .

# Expose the port Gunicorn will listen on
EXPOSE 10000

# Health check Cloud Run uses this to detect readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:10000/api/health || exit 1

# Production server: 2 workers, 4 threads each, 120s timeout
# (model inference + Gemini call can take up to ~10s; generous timeout)
CMD ["sh", "-c", "gunicorn run:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 --access-logfile - --error-logfile -"]
