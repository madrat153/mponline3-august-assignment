FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the demo models at image build time so the container is ready to
# serve immediately. Replace with real dataset training before production
# use -- see each script's docstring in training/.
RUN python training/train_product_classifier.py \
    && python training/train_face_db.py \
    && python training/train_sentiment_model.py \
    && python training/train_chatbot_model.py

ENV SMART_RETAIL_API_KEY=dev-demo-key-123
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
