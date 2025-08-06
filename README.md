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

```bash
git clone https://github.com/EntropyWeaver/fashion-assistant.git
cd fashion-advisor
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate
pip install -r requirements.txt

# Start backend
uvicorn main:app --reload

# In another terminal, start Streamlit
streamlit run streamlit_app.py
```

You can also run `run_app.bat` (Windows only) for one-click startup with timeout syncing.

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

See [📘 ](./extra_notes.md)[`Extra Notes`](./extra_notes.md) for a deep dive into:

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

### ⚠️ Guardrail with Cached Keyword Filtering

Before processing the user input, the application checks for offensive language 
using a locally cached list of banned keywords.

We use `functools.lru_cache` to load the list once from `ban_kwds.json`, making 
the validation extremely fast even if called repeatedly.

This helps block inappropriate requests early in the flow without involving any 
external API or image processing logic.

### 🛡️ Offensive Language Guardrail

To ensure respectful interaction with the assistant, we implemented a local filtering mechanism before any request is processed. This prevents inappropriate prompts from being sent to the LLM or triggering the similarity search engine.

**Highlights:**
- Banned words are loaded from a JSON file using `@lru_cache` for efficiency.
- The guardrail checks user input immediately upon receiving the request.
- If offensive content is detected, a refusal message is returned without processing the image or calling the LLM.
- Multilingual support (EN/ES) with tailored refusal messages.

**Backend logic:**
```python
if contains_offensive_language(text):
    return JSONResponse({"answer": "refusal"})

### 🛡️ Guardrail Architecture

1. **KeywordGuard** – cached set lookup (O(1))
2. **ToxicityGuard** – Detoxify ML model for nuanced insults
3. **OpenAI Moderation** – optional last-resort
4. **RateLimiter** – blocks abusive users (HTTP 429)

Pipeline short-circuits on first positive match, saving compute and tokens.