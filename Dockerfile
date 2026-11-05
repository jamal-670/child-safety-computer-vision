FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/

RUN mkdir -p models \
    && python -c "from urllib.request import urlretrieve; urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt', 'models/yolov8n-pose.pt')"

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--headless"]
