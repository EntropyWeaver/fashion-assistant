"""One end-to-end request with real CLIP retrieval and one OpenAI API call.

Run from the repository root: python -m scripts.smoke_live
This script incurs API usage. It is deliberately excluded from pytest.
"""

import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai import OpenAI

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the live smoke test")
    from app.api.main import app

    sample = Path("data/dresses/dress1.jpg")
    with patch(
        "app.services.fashion_advisor.OpenAI",
        side_effect=lambda **kwargs: OpenAI(max_retries=0, timeout=30, **kwargs),
    ):
        response = TestClient(app).post(
            "/query",
            data={"text": "Describe this garment briefly.", "k": "1", "lang": "en"},
            files={"image": (sample.name, sample.read_bytes(), "image/jpeg")},
        )
    response.raise_for_status()
    result = response.json()
    assert result["similar_images"] and result["answer"].strip()
    assert not list(Path("temp").glob("*.jpg"))
    print("Live smoke OK: one retrieved garment and a non-empty English response")


if __name__ == "__main__":
    main()
