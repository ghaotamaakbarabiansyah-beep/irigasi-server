FROM python:3.11-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Run FastAPI with uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]