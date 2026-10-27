FROM python:3.14-slim

# By default docker containers run as root.
# Create a non-root user with specific UID and GID.
# "Principle of least privilege"
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install curl for Docker healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to /app
WORKDIR /app

# Pythons stdout and stderr are buffered by default.
# With this flag, logs are sent straight to terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# UV optimizations
# Compile bytecode .py to .pyc to speed up containers startup.
# Copy dependencies instead of symlinking them to speed up.
# Do not install dev dependencies to reduce image size.
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_NO_CACHE=1

# gives non-root user access to its own cache
ENV HF_HOME=/home/nonroot/.cache/huggingface

# set the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Ensure non-root user owns necessary runtime directories
RUN mkdir -p /home/nonroot/.cache /app/data/scraping /app/weights \
 && chown -R nonroot:nonroot /home/nonroot /app/data /app/weights

# Install dependencies first - better chache behavior.
COPY --chown=nonroot:nonroot pyproject.toml uv.lock* ./
RUN uv sync --locked --no-install-project

# Install the project files
COPY --chown=nonroot:nonroot README.md app.py ./
COPY --chown=nonroot:nonroot src/ ./src/
COPY --chown=nonroot:nonroot data/ ./data/
COPY --chown=nonroot:nonroot weights/ ./weights/

RUN uv sync --locked

# Switch to non-root user
USER nonroot

# Expose API and Streamlit ports
EXPOSE 8000 8501

ENTRYPOINT []

# Run the API
CMD ["uvicorn", "literature_recommender.api.main:app", "--host", "0.0.0.0", "--port", "8000"]