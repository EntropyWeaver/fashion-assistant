from pathlib import Path
from unittest.mock import Mock, patch

from app.services.fashion_advisor import query_fashion_advisor


def test_multimodal_request_contains_image_and_selected_language(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    image = Path(__file__).resolve().parents[1] / "data/dresses/dress1.jpg"
    mock_client = Mock()
    mock_client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="Pair it with a jacket."))
    ]

    with patch("app.services.fashion_advisor.OpenAI", return_value=mock_client):
        answer = query_fashion_advisor([str(image)], "What goes with this?", lang="en")

    assert answer == "Pair it with a jacket."
    payload = mock_client.chat.completions.create.call_args.kwargs
    assert payload["model"] == "gpt-4.1-nano"
    assert "English" in payload["messages"][0]["content"]
    content = payload["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "What goes with this?"}
    assert any(block.get("image_url", {}).get("url", "").startswith("data:image/jpeg;base64,") for block in content)
