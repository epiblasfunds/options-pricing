FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements del dashboard
COPY src/dashboard/requirements.txt requirements.txt

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar solo el código fuente necesario
COPY src/ src/

ENV PORT=8080

CMD ["gunicorn", "dashboard.main:server", "--chdir", "src"]