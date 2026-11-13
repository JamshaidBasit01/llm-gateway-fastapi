FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[redis]" \
    && useradd --create-home --uid 10001 gateway \
    && mkdir /data && chown gateway /data

USER gateway
ENV DB_PATH=/data/gateway_usage.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "--factory", "llm_gateway.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
