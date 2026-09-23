from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from app.retrieval.engine import RetrievalEngine


def test_catalog_indexes_and_returns_existing_images_without_model_download():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.encode_image.side_effect = lambda tensor: tensor

    def preprocess(image):
        mean_rgb = np.asarray(image, dtype="float32").mean(axis=(0, 1)) / 255
        return torch.from_numpy(mean_rgb)

    with patch("app.retrieval.engine.open_clip.create_model_and_transforms", return_value=(model, None, preprocess)):
        engine = RetrievalEngine(str(data_dir), device="cpu")

    results = engine.search(str(data_dir / "dresses/dress1.jpg"), k=3, threshold=-1)
    assert engine.index.ntotal == 9
    assert len(results) == 3
    assert all(Path(item["image"]).is_file() for item in results)
    assert results[0]["image"] == str(data_dir / "dresses/dress1.jpg")
