FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY requirements-base.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 600 -r requirements-base.txt

FROM base AS inference
COPY requirements-inference.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 600 -r requirements-inference.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS stream
COPY requirements-stream.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 600 -r requirements-stream.txt
COPY . .
CMD ["python", "-m", "src.streaming.consumer"]

FROM base AS producer
COPY requirements-producer.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 600 -r requirements-producer.txt
COPY . .
CMD ["python", "-m", "src.datagen.producer"]

FROM base AS dashboard
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 600 -r requirements-dashboard.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
