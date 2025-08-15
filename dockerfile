# syntax=docker/dockerfile:1
# Streamlit + PyDeck app container for Kubernetes

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (curl for healthcheck, and common CA certs)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

# Install Python deps first for better caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -g 10001 appgroup \
	&& useradd -m -u 10001 -g appgroup appuser \
	&& chown -R appuser:appgroup /app
USER appuser

# Streamlit default port
EXPOSE 8501

# Healthcheck uses Streamlit internal health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["python", "-m", "streamlit", "run", "main_pydeck.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]

