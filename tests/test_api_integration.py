"""Exercise the HTTP boundary without downloading CLIP weights or calling OpenAI."""

import importlib
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "locals").mkdir()
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "locals/ban_kwds.json",
        tmp_path / "locals/ban_kwds.json",
    )
    with patch("app.retrieval.engine.RetrievalEngine") as engine_class:
        engine = engine_class.return_value
        engine.search.return_value = [{"image": "data/dresses/dress1.jpg"}]
        module = importlib.import_module("app.api.main")
        app = module.create_app()
    with patch.object(module, "query_fashion_advisor", return_value="A nice match") as advisor:
        yield TestClient(app), engine, advisor, tmp_path


def test_upload_uses_private_temp_file_and_cleans_it(api):
    client, engine, advisor, workdir = api
    image_bytes = (Path(__file__).resolve().parents[1] / "data/dresses/dress1.jpg").read_bytes()
    response = client.post(
        "/query",
        data={"text": "What matches this?", "lang": "en", "k": "2"},
        files={"image": ("../../escaped.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json() == {"similar_images": ["data/dresses/dress1.jpg"], "answer": "A nice match"}
    stored_path = Path(engine.search.call_args.args[0]).resolve()
    assert stored_path.parent == (workdir / "temp").resolve()
    assert stored_path.name != "escaped.jpg"
    assert not stored_path.exists()
    assert not (workdir.parent / "escaped.jpg").exists()
    advisor.assert_called_once_with(["data/dresses/dress1.jpg"], "What matches this?", lang="en")


def test_temp_file_is_removed_if_advisor_fails(api):
    client, engine, advisor, workdir = api
    advisor.side_effect = RuntimeError("API unavailable")
    image_bytes = (Path(__file__).resolve().parents[1] / "data/tops/top1.jpg").read_bytes()
    response = client.post(
        "/query", data={"text": "Do these match?"},
        files={"image": ("top1.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 500
    assert not list((workdir / "temp").iterdir())


def test_refusal_short_circuits_retrieval_and_model(api):
    client, engine, advisor, workdir = api
    response = client.post(
        "/query", data={"text": "qué mierda"},
        files={"image": ("dress.jpg", b"any bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json() == {"answer": "refusal"}
    engine.search.assert_not_called()
    advisor.assert_not_called()
    assert not (workdir / "temp").exists()
