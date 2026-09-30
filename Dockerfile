FROM python:3.12-slim

WORKDIR /app

# Sirf API ki dependencies install karo
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

# API aur trained model copy karo
COPY app ./app
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]