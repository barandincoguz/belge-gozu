FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --extra ml --no-emit-project -o /tmp/req.txt && uv pip install --system --no-cache -r /tmp/req.txt && rm -f /tmp/req.txt
COPY src ./src
RUN uv pip install --system --no-cache --no-deps .
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
EXPOSE 7860
CMD ["belge-gozu", "serve", "--pull", "--port", "7860"]
