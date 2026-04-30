# ============================================================================
# DAAP — Production Dockerfile
# ============================================================================
# Build:  docker build -t daap:latest .
# Run:    docker compose up -d
# ============================================================================

FROM python:3.12-slim AS base

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Tell pydantic to use pure-Python mode (avoids jiter Rust DLL issues
    # on some kernels / App Control policies).
    PYDANTIC_PURE_PYTHON=1 \
    # Playwright stores browsers here; stays outside /app so the
    # application volume mount doesn't hide it.
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    # DAAP persistent data directory (SQLite DBs, etc.)
    DAAP_DATA_DIR=/data

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 1: system packages
# ---------------------------------------------------------------------------
# Keep build tools alive through the pip install + playwright install so that
# wheels with native extensions (lxml, aiohttp, etc.) compile successfully.
# We strip them in a separate RUN after compilation to shrink the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        # --- Compiler / native build ---
        build-essential \
        # --- lxml / crawl4ai native deps ---
        libxml2-dev \
        libxslt1-dev \
        libffi-dev \
        # --- Chromium runtime requirements (Playwright) ---
        libnss3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        # --- General utilities ---
        curl \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 2: Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt .

# Install Python packages first (better layer caching — only re-runs when
# requirements.txt changes, not on every source change).
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3: Playwright browsers
# ---------------------------------------------------------------------------
# Must run AFTER pip install (playwright Python package must exist first).
# --with-deps installs the OS-level Chromium libraries into PLAYWRIGHT_BROWSERS_PATH.
RUN python -m playwright install --with-deps chromium

# ---------------------------------------------------------------------------
# Stage 4: crawl4ai post-install setup
# ---------------------------------------------------------------------------
# crawl4ai ships a CLI setup step that downloads browser driver shims and
# verifies the Playwright installation.  Run non-interactively.
RUN python -m crawl4ai.cli setup --skip-playwright 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 5: strip build tooling
# ---------------------------------------------------------------------------
# Removes gcc/g++/make etc. that were only needed for wheel compilation.
# libnss3 and other Chromium runtime libs are intentionally kept.
RUN apt-get purge -y --auto-remove \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# ---------------------------------------------------------------------------
# Stage 6: application source
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# Stage 7: non-root runtime user
# ---------------------------------------------------------------------------
# /data  — persistent SQLite databases (mounted as a Docker volume)
# /opt/playwright — Chromium binaries (must be readable + executable by daap)
RUN useradd --system --uid 1001 --home /home/daap --create-home daap \
 && mkdir -p /data \
 && chown -R daap:daap /app /data /opt/playwright /home/daap

USER daap

# ---------------------------------------------------------------------------
# Ports & health
# ---------------------------------------------------------------------------
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
# tini reaps zombie processes produced by Playwright/Chromium child processes.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "exec uvicorn daap.main:app --host 0.0.0.0 --port ${PORT:-8000} --ws wsproto --workers ${WEB_CONCURRENCY:-1} --proxy-headers --forwarded-allow-ips '*' --log-level ${LOG_LEVEL:-info}"]
