"""
Retrieval engine for fashion images using CLIP and FAISS.

This module encapsulates the logic needed to compute image embeddings with
OpenCLIP, build a FAISS index over those embeddings and perform nearest
neighbour queries.  The engine is initialised with a root directory
containing image subfolders (e.g., ``tops``, ``dresses``) and lazily
constructs the index on first use.  Subsequent search calls will reuse
the precomputed index and label mapping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np
import pandas as pd
import faiss
import torch
import open_clip
from PIL import Image
from io import BytesIO
import base64

class RetrievalEngine:
    """
    Engine responsible for loading images, extracting embeddings, building
    a FAISS index and executing similarity searches.

    Parameters
    ----------
    data_root : str
        The root directory containing subfolders of images. Each
        subfolder name will be treated as the category of its images.
    device : str, optional
        The torch device on which to run the embedding model. Defaults
        to ``cuda`` if available, otherwise ``cpu``.
    model_name : str, optional
        Name of the CLIP model to load via open_clip.  Defaults to
        ``ViT-B-32``.
    pretrained : str, optional
        Which pretrained weights to load for the model.  Defaults to
        ``laion400m_e32``.
    """

    def __init__(
        self,
        data_root: str,
        device: str | None = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion400m_e32",
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists() or not self.data_root.is_dir():
            raise ValueError(f"El directorio de datos '{data_root}' no existe o no es un directorio")

        # Determine device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and preprocessing once.  open_clip returns a tuple
        # (model, _, preprocess).
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()

        # Placeholders for index and labels
        self.index: faiss.Index | None = None
        self.labels: pd.DataFrame | None = None

        # Build index immediately.  This call populates self.index and
        # self.labels.
        self._build_index()

    def _get_embedding(self, image_path: str) -> np.ndarray:
        """Compute a normalized embedding for an image located at ``image_path``.

        Parameters
        ----------
        image_path : str
            Path to the image on disk.

        Returns
        -------
        np.ndarray
            A 2D numpy array with shape (1, embedding_dim) and dtype
            ``float32``.  The embedding is L2-normalised.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ No se pudo abrir la imagen: {image_path} → {e}")
            raise
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(image_tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    def _load_dataset_images(self) -> tuple[np.ndarray, List[tuple[str, str]]]:
        """Traverse the dataset root and compute embeddings for every image.

        Returns
        -------
        embeddings : np.ndarray
            Stacked embeddings with shape (N, D) where N is the number of
            images.
        labels : list of (str, str)
            A list of (filename, category) pairs corresponding to each
            embedding.  The filename is an absolute path.
        """
        embeddings: List[np.ndarray] = []
        labels: List[tuple[str, str]] = []
        # Iterate over immediate subdirectories.  Each directory is a
        # category.
        for category_dir in self.data_root.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            for image_file in category_dir.glob("*.jpg"):
                path = str(image_file)
                try:
                    emb = self._get_embedding(path)
                    embeddings.append(emb)
                    labels.append((path, category))
                except Exception as exc:
                    # Skip problematic files and continue processing
                    print(f"Error al procesar {path}: {exc}")
        if not embeddings:
            raise RuntimeError(f"No se encontraron imágenes en {self.data_root}")
        embeddings_np = np.vstack(embeddings)
        return embeddings_np, labels

    def _build_index(self) -> None:
        """Build the FAISS index and populate the labels DataFrame."""
        # Load dataset and compute embeddings
        embeddings, labels = self._load_dataset_images()
        # Build FAISS index
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        # Save index and labels
        self.index = index
        self.labels = pd.DataFrame(labels, columns=["filename", "category"])

    def search(self, query_image_path: str, k: int = 5, threshold: float = 0.6) -> List[Dict[str, object]]:
        """Find the top ``k`` most similar images for a query image exceeding a similarity threshold.

        Parameters
        ----------
        query_image_path : str
            Path to the image to be queried.
        k : int, default=5
            Number of similar images to return.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing the keys ``image``
            (absolute file path), ``category`` and ``similarity`` (cosine
            similarity in embedding space rounded to four decimals).
        """
        if self.index is None or self.labels is None:
            raise RuntimeError("El índice aún no ha sido construido")
        
        # Compute embedding for the query image
        emb = self._get_embedding(query_image_path)
        # Perform the search.  FAISS returns D and I arrays where D
        # contains similarities and I contains indices into the labels
        # DataFrame.  Note: FAISS expects shape (1, D) arrays.
        D, I = self.index.search(emb, k,)
        results: List[Dict[str, object]] = []
        for idx, dist in zip(I[0], D[0]):
            row = self.labels.iloc[idx]
            if dist >= threshold:
                results.append(
                    {
                        "image": row["filename"],
                        "category": row["category"],
                        "similarity": round(float(dist), 4),
                    }
                )
        return results
        
    