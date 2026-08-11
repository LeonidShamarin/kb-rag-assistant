FROM python:3.11-slim

# Hugging Face Spaces запускає Docker-контейнер від НЕ-root користувача з uid 1000.
# Якщо лишити /app власністю root, `ingest` на старті впаде з Permission denied при
# спробі записати index/ — і Space покаже «Runtime error» без зрозумілої причини.
# Тому користувач створюється явно, а не успадковується від образу.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src/ src/
COPY --chown=user web/ web/
COPY --chown=user data/ data/
COPY --chown=user eval/ eval/
COPY --chown=user main.py .

# Індекс будується на старті контейнера, а не при збірці: інакше в образ
# потрапили б вектори, прив'язані до конкретного ключа й версії моделі.
RUN mkdir -p index output

# 7860 — порт, який Hugging Face Spaces очікує від Docker-контейнера.
ENV PORT=7860
EXPOSE 7860

# Порт береться з $PORT, а не зашитий: інакше healthcheck мовчки перевіряв би не
# той сервіс, щойно порт змінили через оточення.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT','7860')}/health\").status==200 else 1)"

# GEMINI_API_KEY передавати через `docker run -e ...`, --env-file .env або
# Settings → Variables and secrets у Space. У репозиторії ключа немає ніколи.
ENTRYPOINT ["/bin/sh", "-c", "python main.py ingest && exec python main.py serve \"$@\"", "--"]
