FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first (faster rebuilds)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Set Python path
ENV PYTHONPATH=src

EXPOSE 8000

CMD ["uvicorn", "fraud.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
