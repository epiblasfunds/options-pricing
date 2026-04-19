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


# Copiar el código fuente
COPY src/ src/
COPY resources/ /app/resources/


# Expone el puerto estándar de Cloud Run
EXPOSE 8080
ENV PORT=8080

CMD ["gunicorn", "dashboard.main:server", "--chdir", "src", "--bind", ":8080", "--workers", "2"]