# ─────────────────────────────────────────────────────────────────────────────
# AI Agent Lab — Dockerfile
# Multi-stage build: builder installs deps, runtime is lean.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Prevents .pyc files and forces stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build tools needed by some packages (numpy, sympy wheels usually
# come pre-compiled, but pint/backoff compile clean).
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (layer cache)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=ai_agent_lab.settings \
    PORT=8000

# Create non-root user for security
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup . .

# Collect static files (no GROQ key needed for this step)
RUN python manage.py collectstatic --noinput --settings=ai_agent_lab.settings 2>/dev/null || true

# Switch to non-root
USER appuser

# Django default port
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/')" || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Uvicorn ASGI server — handles Django's async views natively
CMD ["sh", "-c", \
     "python manage.py migrate --noinput && \
      uvicorn ai_agent_lab.asgi:application \
        --host 0.0.0.0 \
        --port ${PORT} \
        --workers 2 \
        --log-level info"]
