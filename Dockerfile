# QuantGrid image: one image, four roles (migrate | api | coordinator | worker).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    # one BLAS/OpenMP thread per worker process: parallelism comes from worker processes
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN pip install --no-cache-dir "uv==0.11.32"
WORKDIR /app

# Dependencies first (cached layer), resolved exactly from the lockfile.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra llm

COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src
RUN uv sync --frozen --no-dev --extra llm

ARG GIT_COMMIT=unknown
ENV PATH=/opt/venv/bin:$PATH \
    QG_GIT_COMMIT=$GIT_COMMIT \
    QG_DATA_DIR=/var/lib/quantgrid/data \
    QG_ARTIFACT_ROOT=/var/lib/quantgrid/artifacts

RUN useradd --create-home --uid 10001 quantgrid \
    && mkdir -p /var/lib/quantgrid \
    && chown -R quantgrid:quantgrid /var/lib/quantgrid
USER quantgrid

EXPOSE 8000
ENTRYPOINT ["quantgrid"]
CMD ["api", "--host", "0.0.0.0", "--port", "8000"]
