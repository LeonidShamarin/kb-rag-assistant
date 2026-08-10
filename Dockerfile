FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY web/ web/
COPY data/ data/
COPY eval/ eval/
COPY main.py .

# Індекс будується на старті контейнера, а не при збірці: інакше в образ
# потрапили б вектори, прив'язані до конкретного ключа й версії моделі.
RUN mkdir -p index output

# 7860 — порт, який Hugging Face Spaces очікує від Docker-контейнера.
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

# GEMINI_API_KEY передавати через `docker run -e ...` або --env-file .env
ENTRYPOINT ["/bin/sh", "-c", "python main.py ingest && exec python main.py serve \"$@\"", "--"]
