"""
Utility functions for image encoding.

This module contains helper functions to convert image files into base64
encoded strings suitable for inclusion in OpenAI API requests as data
URLs.  The functions here should be imported by any service that needs
to transmit image content to a language model.
"""

from __future__ import annotations

import base64
from pathlib import Path


def encode_image_to_base64(image_path: str | Path) -> str:
    """Read an image from disk and return its contents encoded in base64.

    Parameters
    ----------
    image_path : str or Path
        The absolute or relative path to the image file.

    Returns
    -------
    str
        A base64 encoded string of the image's binary data.  The caller
        should prepend the appropriate MIME type prefix (e.g.
        ``"data:image/jpeg;base64,"``) when constructing a data URL.

    Raises
    ------
    FileNotFoundError
        If the image path does not exist.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No se encuentra la imagen: {image_path}")
    with image_path.open("rb") as f:
        binary_data = f.read()
    encoded = base64.b64encode(binary_data).decode("utf-8")
    return encoded
