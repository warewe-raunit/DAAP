FROM python:3.12-slim

WORKDIR /app

# lxml build deps + base utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install playwright + all chromium system deps (required by crawl4ai DeepCrawl tool)
RUN python -m playwright install --with-deps chromium

COPY . .

EXPOSE 8000

CMD ["uvicorn", "daap.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
