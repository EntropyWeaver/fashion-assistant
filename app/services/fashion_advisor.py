"""
Service for interacting with the GPT‑4o model using multimodal prompts.

This module exposes a single function, :func:`query_fashion_advisor`,
which takes an image path and a text prompt from the user.  The image
is encoded to base64 and sent along with the text as part of a
multimodal request to the GPT‑4o model via the OpenAI API.  The
function returns the assistant's reply as a string.  The API key is
retrieved from the ``OPENAI_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
from typing import Tuple

import openai

from app.utils.image_encoding import encode_image_to_base64


def query_fashion_advisor(image_path: str, user_prompt: str, model: str = "gpt-4o") -> str:
    """Query the GPT‑4o model with a multimodal prompt consisting of text and an image.

    Parameters
    ----------
    image_path : str
        Path to the image that should be analysed by the assistant.
    user_prompt : str
        The user's natural language question or instruction.
    model : str, default="gpt-4o"
        The name of the OpenAI model to query.

    Returns
    -------
    str
        The assistant's reply.  Any leading or trailing whitespace is
        stripped.  If the API key is not configured an exception will
        propagate to the caller.
    """
    # Retrieve the API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La variable de entorno OPENAI_API_KEY no está configurada. Establece tu clave de API de OpenAI."
        )

    # Encode the image into a data URL
    encoded_image = encode_image_to_base64(image_path)
    # For simplicity we assume JPEG images; adjust the MIME type if
    # necessary based on the actual image extension.
    data_url = f"data:image/jpeg;base64,{encoded_image}"

    # Prepare messages for the chat API.  The OpenAI python library
    # accepts a list of messages where the `content` field can be either a
    # string or a list of content parts.  When providing multimodal
    # messages, we supply a list with dictionaries describing the type of
    # each part.
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un estilista virtual experto en moda. Analiza la prenda "
                "en la imagen y proporciona consejos de estilo personalizados "
                "respondiendo siempre en español."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    # Initialize the OpenAI client.  openai==1.x supports a Client class
    # but also offers module-level functions.  We use the module-level
    # interface here for simplicity.
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
    except Exception as exc:
        raise RuntimeError(f"Error al llamar al API de OpenAI: {exc}")

    # Extract and return the assistant's message content
    reply = response.choices[0].message.get("content", "").strip()
    return reply
