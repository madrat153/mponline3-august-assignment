author :-Ratnesh Dixit
# AI-Powered Smart Retail & Customer Intelligence Platform

A single deployable system combining **computer vision** (face recognition
for returning-customer detection, product image classification), **NLP**
(customer review sentiment analysis), and a **hybrid FAQ chatbot**, all
exposed through one production-style FastAPI service.

This implementation follows the Major Project brief module-for-module —
see [Module coverage](#module-coverage) below.

## Quick start

```bash
git clone <this-repo>
cd smart-retail-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train all four demo models (uses the sample/synthetic data shipped in data/)
python training/train_product_classifier.py
python training/train_face_db.py
python training/train_sentiment_model.py
python training/train_chatbot_model.py

# Run the API
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger UI.

Every endpoint except `/health` and `/` requires an API key header:

```
X-API-Key: dev-demo-key-123
```

Change it via `SMART_RETAIL_API_KEY` in your environment — see `.env.example`.

## Run with Docker

```bash
docker compose up --build
```

This builds the image, trains all demo models at build time, and serves
the API on `http://localhost:8000`.

## Run the tests

```bash
pytest -v
```

All 14 tests pass out of the box against the shipped demo models.

## API endpoints

| Method | Path                 | Description                                                |
|--------|----------------------|-------------------------------------------------------------|
| POST   | `/recognize-face`    | Upload an image → returns customer status (new/returning)   |
| POST   | `/register-face`     | Upload an image → enrolls a new customer face template      |
| POST   | `/classify-product`  | Upload a product photo → returns predicted category         |
| POST   | `/analyze-sentiment` | Send text → returns positive/negative/neutral + scores      |
| POST   | `/chatbot`           | Send a message → returns a FAQ bot reply                    |
| GET    | `/dashboard/stats`   | Aggregate visit / sentiment / chat stats (JSON)              |
| GET    | `/health`            | Unauthenticated health check                                  |

### Example: curl

```bash
curl -X POST http://localhost:8000/analyze-sentiment \
  -H "X-API-Key: dev-demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"text": "The delivery was fast and the quality is amazing!"}'

curl -X POST http://localhost:8000/chatbot \
  -H "X-API-Key: dev-demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your return policy?"}'

curl -X POST http://localhost:8000/classify-product \
  -H "X-API-Key: dev-demo-key-123" \
  -F "file=@/path/to/product.jpg"
```

## Folder structure

```
smart-retail-ai/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py                # settings / paths
│   ├── schemas.py                # Pydantic request/response models
│   ├── security.py               # API key auth dependency
│   ├── routers/
│   │   ├── vision.py             # /recognize-face, /register-face, /classify-product
│   │   ├── nlp.py                # /analyze-sentiment
│   │   ├── chatbot.py            # /chatbot
│   │   └── dashboard.py          # /dashboard/stats
│   ├── services/
│   │   ├── cv_utils.py           # Module A1 deliverable
│   │   ├── cv_service.py         # Module A2 + A3
│   │   ├── text_utils.py         # Module B1 deliverable
│   │   ├── nlp_service.py        # Module B2
│   │   ├── chatbot_service.py    # Module B3
│   │   ├── dashboard_service.py  # stats aggregation
│   │   └── pipeline.py           # Module C1: unified, load-once pipeline
│   └── models/                   # trained model artifacts land here
├── training/
│   ├── train_product_classifier.py       # default: HOG + SVM
│   ├── train_product_classifier_tf.py    # stretch goal: MobileNetV2
│   ├── train_face_db.py
│   ├── train_sentiment_model.py
│   └── train_chatbot_model.py
├── notebooks/                    # thin wrappers around the scripts above
├── data/
│   ├── reviews.csv               # sample sentiment training data
│   └── intents.json              # 25 custom chatbot intents
├── tests/
│   └── test_endpoints.py         # 14 passing smoke tests
├── runtime_logs/                 # created at runtime: visits/reviews/chat logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/deploy.yml  # lint + test + build CI
```

## Module coverage

| Brief module                          | Implementation |
|----------------------------------------|----------------|
| A1. OpenCV basics                      | `app/services/cv_utils.py` |
| A2. Image classification               | `app/services/cv_service.py::ProductClassifierService`, `training/train_product_classifier.py` (+ optional MobileNetV2 in `train_product_classifier_tf.py`) |
| A3. Face recognition fundamentals      | `app/services/cv_service.py::FaceRecognitionService`, `training/train_face_db.py` |
| B1. Text preprocessing                 | `app/services/text_utils.py` |
| B2. Sentiment analysis                 | `app/services/nlp_service.py`, `training/train_sentiment_model.py` |
| B3. Chatbot (rule + ML hybrid)         | `app/services/chatbot_service.py`, `training/train_chatbot_model.py`, `data/intents.json` |
| C1. Unified pipeline                   | `app/services/pipeline.py` |
| C2. Serialization                      | `joblib` for sklearn models, LBPH `.yml` + `pickle` for face encodings |
| C3. API layer (FastAPI + Swagger)      | `app/main.py`, `app/routers/*` |
| C4. Deployment                         | `Dockerfile`, `docker-compose.yml`, API-key auth, `.github/workflows/deploy.yml` |

## Important dataset & model notes

To keep this repository runnable **completely offline** (no dataset
downloads, no GPU required), the shipped training scripts generate small
**synthetic/demo datasets** for the product classifier and face database,
and a small **generated sample CSV** (~84 rows) for sentiment. Everything
wires up end-to-end so you can run and demo the full platform
immediately — but because these are tiny toy datasets, don't expect
production-grade accuracy (e.g. the sentiment model can misclassify text
that uses vocabulary outside its ~84-row training set). For an actual
graded submission, swap in real data:

- **Product images**: Fashion-MNIST or Kaggle's "Retail Product Checkout
  Dataset" (see `training/train_product_classifier.py` docstring). For
  higher accuracy, use `training/train_product_classifier_tf.py`
  (MobileNetV2 transfer learning) instead of the default HOG+SVM baseline.
- **Face recognition**: enroll real, consenting faces via
  `POST /register-face` instead of relying on `training/train_face_db.py`'s
  synthetic seed data (see its docstring), or use an LFW subset for practice.
- **Reviews/sentiment**: Kaggle "Women's E-Commerce Clothing Reviews" or
  an Amazon Reviews subset — same two-column `review_text,sentiment` schema.
- **Chatbot intents**: `data/intents.json` already contains 25 original,
  hand-written intents covering common retail FAQs — real, usable data,
  not a placeholder.

## Ethics note: facial recognition in retail

Facial recognition for "returning customer" detection raises real
consent, privacy, and bias concerns and should not be deployed against
real customers without:

- **Explicit opt-in consent** (e.g. a loyalty-program sign-up flow, not
  covert in-store capture).
- A **clear retention and deletion policy** for stored face encodings.
- **Bias/accuracy audits** across demographic groups before launch, since
  face-recognition systems have documented accuracy gaps for some
  skin tones, ages, and genders.
- Compliance with local biometric-data regulations (e.g. BIPA in
  Illinois, GDPR in the EU) before any production use.

## Stretch goals (see brief section 7)

- Fine-tune DistilBERT for sentiment instead of TF-IDF + Logistic Regression.
- Add a WebSocket endpoint for real-time video-stream face recognition.
- Log prediction-confidence drift over time for model monitoring.
- Build a Streamlit dashboard on top of `GET /dashboard/stats`.
- A/B test two chatbot response strategies and compare user ratings.
