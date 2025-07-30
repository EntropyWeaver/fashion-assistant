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
import mimetypes
from typing import List
from openai import OpenAI
from load_dotenv import load_dotenv

from app.utils.image_encoding import encode_image_to_base64

load_dotenv()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

def get_data_url(image_path: str) -> str:
    """Codifica una imagen como data URL si su tipo MIME está permitido.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Formato de imagen no soportado: {mime_type}")

    encoded = encode_image_to_base64(image_path)
    return f"data:{mime_type};base64,{encoded}"



def query_fashion_advisor(image_paths: List[str],
                          user_prompt: str,
                          model: str = "gpt-4o") -> str:
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
    # Coding all valid imgsas URLs data
    image_blocks = []
    for path in image_paths:
        try:
            data_url = get_data_url(path)
            image_blocks.append({"type": "image_url",
                                 "image_url": {"url": data_url}})
        except Exception as e:
            print(f"No se pudo procesar {path}: {e}")  
    
    if not image_blocks:
        raise ValueError("None of images are valid.")  
        
    client = OpenAI(api_key=api_key)
    # Encode the image into a data URL
    
    # For simplicity we assume JPEG images; adjust the MIME type if
    # necessary based on the actual image extension.
    

    # Prepare messages for the chat API.  The OpenAI python library
    # accepts a list of messages where the `content` field can be either a
    # string or a list of content parts.  When providing multimodal
    # messages, we supply a list with dictionaries describing the type of
    # each part.
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un estilista virtual experto en moda. Analiza las prendas "
                "en las imágenes proporcionadas y responde siempre en español con consejos "
                "personalizados de estilo y combinación de ropa. Ignora preguntas fuera del ámbito de moda."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": "Las siguientes imágenes representan prendas similares encontradas."},
                *image_blocks
            ],
        },
    ]

    # Initialize the OpenAI client.  openai==1.x supports a Client class
    # but also offers module-level functions.  We use the module-level
    # interface here for simplicity.
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
    except Exception as exc:
        raise RuntimeError(f"Error al llamar al API de OpenAI: {exc}")

    # Extract and return the assistant's message content
    reply = (response.choices[0].message.content or "").strip()
    return reply
