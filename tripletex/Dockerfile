FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV TRIPLETEX_RUNTIME=container

COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY README.md README.md

RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src.tripletex.main:app", "--host", "0.0.0.0", "--port", "8080"]
