FROM python:3.12-slim

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# CPU torch wheels come from PyTorch's index; --extra-index-url lets pip find them
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY main.py .
COPY best.pt .

USER appuser

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/' % os.environ.get('PORT','8000'))" || exit 1

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}