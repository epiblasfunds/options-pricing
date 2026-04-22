FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/dashboard/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY resources/ ./resources

EXPOSE 8080

CMD ["gunicorn", "dashboard.main:server", "--chdir", "src", "--bind", ":8080", "--workers", "2"]