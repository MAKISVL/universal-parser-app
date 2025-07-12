FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY docker-compose.yml .
COPY Dockerfile . 

COPY templates/ templates/
COPY static/ static/

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]