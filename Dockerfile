FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache ".[ml]"
ENV BG_DEVICE=cpu
EXPOSE 7860
CMD ["belge-gozu", "serve", "--pull", "--port", "7860"]
