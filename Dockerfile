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

# torch ставиться ОКРЕМО і з CPU-індексу PyTorch. Звичайний `pip install torch`
# з PyPI тягне за собою пакети nvidia-cu* на ~1.8 ГБ, які на CPU-Space не
# виконуються жодного разу. Різниця в розмірі образу — 2.5 ГБ проти ~700 МБ,
# і на безкоштовному тірі це різниця між збіркою і таймаутом збірки.
RUN pip install --no-cache-dir --user \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.2"

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Ваги моделі кладемо в образ на етапі збірки, а не качаємо на старті. Інакше
# кожен холодний старт Space (а він засинає після простою) — це ще 470 МБ з
# мережі перед першим запитом, і будь-який збій huggingface.co означає, що
# сервіс не підніметься взагалі.
ENV HF_HOME=$HOME/.cache/huggingface \
    EMBED_PROVIDER=st \
    EMBED_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBED_MODEL')"

COPY --chown=user src/ src/
COPY --chown=user web/ web/
COPY --chown=user data/ data/
COPY --chown=user eval/ eval/
COPY --chown=user main.py .

# Індекс будується на старті контейнера, а не при збірці: інакше в образ
# потрапили б вектори, прив'язані до конкретної версії моделі.
RUN mkdir -p index output

# 7860 — порт, який Hugging Face Spaces очікує від Docker-контейнера.
ENV PORT=7860
EXPOSE 7860

# Порт береться з $PORT, а не зашитий: інакше healthcheck мовчки перевіряв би не
# той сервіс, щойно порт змінили через оточення. start-period — 120 с, бо перший
# старт витрачає час на ingest 13 документів локальною моделлю.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT','7860')}/health\").status==200 else 1)"

# Провайдер embeddings передається В ОБИДВІ команди. Якщо вказати його лише в
# `serve`, індекс збудується дефолтним провайдером, а запити підуть іншим — і
# пошук мовчки поверне сміття замість помилки.
#
# GEMINI_API_KEY передавати через `docker run -e ...`, --env-file .env або
# Settings → Variables and secrets у Space. У репозиторії ключа немає ніколи.
# Без ключа сервіс усе одно піднімається: працює пошук, вимикається генерація.
ENTRYPOINT ["/bin/sh", "-c", "python main.py ingest --embed-provider \"$EMBED_PROVIDER\" --embed-model \"$EMBED_MODEL\" && exec python main.py serve --embed-provider \"$EMBED_PROVIDER\" --embed-model \"$EMBED_MODEL\" \"$@\"", "--"]
