FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra ml --no-install-project

COPY src ./src
COPY scripts/docker_smoke.py ./scripts/docker_smoke.py
RUN uv sync --frozen --no-dev --extra ml \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/index /data/images /data/hf \
    && chown -R 1000:1000 /data /home/app

ARG BG_HF_REVISION="700ac324fffefb22de02c8e90347b31185547948"
ENV BG_HF_REVISION=${BG_HF_REVISION}
ENV BG_HF_DATASET_REPO=barandincoguz/belge-gozu-index
ENV BG_DATA_DIR=/data
ENV BG_INDEX_DIR=/data/index
ENV HF_HOME=/data/hf
ENV BG_DEVICE=cpu
# HERKESE AÇIK DAĞITIM varsayılanları (kütüphane varsayılanları değil — yerelde
# hız sınırı kapalı, sorgu metni loglanır kalır).
#   * hız sınırı: /ask ücretli bir LLM çağrısıdır, /search saf yerel hesaptır —
#     tavanlar da bu orana göre (istemci IP başına, dakikada).
#   * BG_LOG_QUERY_TEXT=false: herkese açık bir demoda yabancıların ham soruları
#     diske yazılmaz; sha256 her koşulda yazıldığı için tekrar/hacim analizi
#     yapılabilir kalır.
ENV BG_RATE_LIMIT_ASK_PER_MIN=10
ENV BG_RATE_LIMIT_SEARCH_PER_MIN=60
ENV BG_LOG_QUERY_TEXT=false

USER 1000:1000
EXPOSE 7860
CMD ["belge-gozu", "serve", "--pull", "--port", "7860"]
