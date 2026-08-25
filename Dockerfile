FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --extra ml -o /tmp/req.txt && uv pip install --system --no-cache -r /tmp/req.txt
COPY src ./src
RUN uv pip install --system --no-cache --no-deps .
ENV BG_DEVICE=cpu
EXPOSE 7860
CMD ["belge-gozu", "serve", "--pull", "--port", "7860"]
