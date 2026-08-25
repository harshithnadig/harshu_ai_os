# Use official lightweight Python 3.12 image
FROM python:3.12-slim-bookworm

# Install uv from official binary image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Set environment variables for performance and clean output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Copy dependency definition files first for efficient layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies into virtual environment without the root project
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code and project metadata
COPY src/ ./src/
COPY README.md ./

# Sync to install the application project itself in non-dev mode
RUN uv sync --frozen --no-dev

# Expose default FastAPI application port
EXPOSE 8000

# Start FastAPI service via Uvicorn
CMD ["uvicorn", "harshu_ai_os.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
