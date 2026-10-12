FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HUB_DATA_DIR=/app/data HUB_VAR_DIR=/app/var
COPY pyproject.toml ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir ".[openai,anthropic]" && python -m agent_hub.seed
RUN useradd --create-home hub && chown -R hub /app
USER hub
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "agent_hub.api:app", "--host", "0.0.0.0", "--port", "8000"]
