# 👗 AI Fashion Advisor — Multimodal Apparel Search & Insights

Discover fashion like never before: this application lets users upload an image of a garment, retrieve visually similar items using neural embeddings, and ask natural-language questions about the uploaded image or its matches — all in a multilingual interface powered by modern LLMs.

---

## 🚀 What Can It Do?

* 📸 Upload clothing images (e.g., a top or dress)
* 🔍 Retrieve similar fashion items from a local dataset using OpenCLIP + FAISS
* 🧠 Ask questions about any selected image — like "What would match this?" or "Is this summer wear?"
* 🌐 Switch between English and Spanish seamlessly
* 📥 Add more fashion items to the dataset simply by copying images into folders

---

## 🧱 Tech Stack

| Layer             | Technology                        |
| ----------------- | --------------------------------- |
| Frontend          | Streamlit                         |
| Backend           | FastAPI                           |
| Embeddings        | OpenCLIP (ViT-B/32)               |
| Similarity Search | FAISS with Cosine Similarity      |
| Language Model    | GPT-4.1 (via OpenAI API)          |
| File Handling     | UUID-named temp storage + cleanup |
| Language Support  | `en.json`, `es.json` localized UI |

---

## 📦 Run Locally

Use Python 3.11: the pinned `faiss-cpu==1.7.4` does not provide a Python 3.12 wheel. Set `OPENAI_API_KEY` in your environment before sending a fashion query; keep it out of Git.

```bash
git clone https://github.com/EntropyWeaver/fashion-assistant.git
cd fashion-assistant
python3.11 -m venv venv # on Windows: py -3.11 -m venv venv
source venv/bin/activate # on Windows cmd: venv\Scripts\activate.bat
pip install -r requirements.txt

# Start backend
uvicorn main:app --reload

# In another terminal, start Streamlit
streamlit run streamlit_app.py
```

You can also run `run_app.bat` (Windows only) for one-click startup with timeout syncing.

## 🧪 Tests

With Python 3.11, install the test dependencies and run the local suite without making API calls:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers the API boundary, file cleanup, multilingual query moderation, message construction, and catalog search with a mocked embedding model. To exercise real OpenCLIP embeddings without an API key, initialize `RetrievalEngine('data')` and search one of the sample garments. The first run downloads model weights.

After setting `OPENAI_API_KEY` for a test project, run `python -m scripts.smoke_live` from the repository root to make one real request with one retrieved image. This is separate from pytest because it uses paid API capacity.

---

## 🧪 Example Usage

1. Upload an image of a garment (e.g., from a website or your camera roll)
2. App returns top 5 similar items (from tops/ or dresses/ folders)
3. Choose which image to ask about (uploaded or retrieved)
4. Ask a question like:

   * *"Is this suitable for a formal dinner?"*
   * *"What kind of jacket would go with this?"*
5. Receive an answer from the LLM, enriched with visual context

---

## 🎯 Real-World Applications

* Personal wardrobe advisors
* Online fashion retail search
* Visual customer support
* Fashion research or trend analysis

---

## 📘 Want the Nerdy Stuff?

See [Extra Notes](./extra_notes.md) for a deep dive into:

* Multimodal prompting via base64
* Cosine similarity vs. L2 in FAISS
* Robustness to irrelevant queries
* Vector retrieval tuning
* Roadmap (Docker, ONNX, vector DBs, personalization)

---

## 🧵 Credits

Built with ❤️ by me, [EntropyWeaver](https://github.com/EntropyWeaver) as a hands-on project exploring RAG, vector search, multimodal prompting and real-world LLM integration. This project represents a real leap from zero to application-ready within days — proof that understanding can evolve fast when you build with purpose.

---

## License

MIT — free to use, adapt, or extend. Fashion belongs to everyone.

### Legacy Code: `assistant.py`

- **Role:** Originally handled LLM prompt construction and offensive language filtering.
- **Deprecated:** The simplified label approach was removed as it degraded performance.
- **Still useful for:** Testing guardrails, generating DataFrames of retrieval results.

### 🛡️ Query moderation

Before saving an uploaded image, searching FAISS or calling OpenAI, the API evaluates the query with Detoxify's **multilingual** model on CPU. The model runs locally and is loaded once on the first query; that first use downloads roughly 1 GB of weights and may be slow. Production deployments should preload the weights. No second OpenAI request is needed for moderation.

The policy in `app/services/query_moderation.py` looks for strong evidence of insults, identity attacks and threats. High general toxicity by itself does not block harsh criticism of clothing, such as “Este vestido es una puta mierda, ¿me enseñas otro?”. It checks clauses separately so an unrelated insult appended to a fashion question can still be caught. A blocked query returns the existing localized refusal. If the classifier is unavailable, the API returns HTTP 503 without contacting FAISS or OpenAI.

These thresholds are starting points measured against a small set of Spanish and English examples. Detoxify was trained for toxic comments and can make mistakes on fashion queries, especially idioms, descriptions of garments and personal distress. Evaluate additional examples before relying on it as a comprehensive safety system. The old word-list code remains in `app/utils/filters.py` for experiments but is no longer part of the API request path.
