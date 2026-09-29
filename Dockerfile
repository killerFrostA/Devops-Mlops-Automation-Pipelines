# Development baseline; pin an approved image digest before production use.
FROM python:3.14-slim AS builder
WORKDIR /build
COPY requirements/build.lock ./requirements/build.lock
RUN python -m pip install --no-cache-dir -r requirements/build.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m build --wheel --no-isolation

FROM python:3.14-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN groupadd --gid 10001 pids && useradd --uid 10001 --gid pids --no-create-home pids
COPY requirements/runtime.lock ./requirements/runtime.lock
RUN python -m pip install --no-cache-dir -r requirements/runtime.lock
COPY --from=builder /build/dist/*.whl /tmp/wheels/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]
CMD ["python", "-m", "src.cli", "serve", "--host", "0.0.0.0"]
