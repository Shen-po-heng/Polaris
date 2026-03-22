# ── Polaris — Local Research Assistant ────────────────────────────────────────
# Multi-stage build keeps the final image lean.
# Runs as a non-root user for security.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Dependencies ───────────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime ────────────────────────────────────────────────────────────────────
FROM base AS runtime

# Create non-root user
RUN addgroup --system polaris \
    && adduser --system --ingroup polaris polaris

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source
COPY --chown=polaris:polaris . /app

USER polaris

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

CMD ["python", "app.py"]
