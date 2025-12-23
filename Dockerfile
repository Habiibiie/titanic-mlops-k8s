# 1. Base Image: Python 3.10
FROM python:3.10-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Set Environment Dependencies
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 4. Copy requirements.txt
COPY requirements.txt .

# 5. Load libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy all files
COPY . .

# 7. Port Settings
EXPOSE 8000

# 8. Start the Code
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

