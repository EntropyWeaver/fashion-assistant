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




def query_fashion_advisor(
    image_paths: List[str],
    user_prompt: str,
    model: str = "gpt-4.1-nano",
    lang: str = "es"
) -> str:
    """Consulta a GPT‑4o con un prompt multimodal formado por texto + imágenes."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La variable de entorno OPENAI_API_KEY no está configurada."
        )

    # Traducción o configuración de instrucciones del sistema por idioma
    system_prompts = {
        "es": (
            "Eres un estilista virtual experto en moda. Analiza las prendas "
            "en las imágenes proporcionadas y responde siempre en español con consejos "
            "personalizados de estilo y combinación de ropa. Ignora preguntas fuera del ámbito de moda."
        ),
        "en": (
            "You are a virtual fashion stylist. Analyze the garments shown in the provided images "
            "and always reply in English with personalized fashion advice and clothing combinations. "
            "Ignore questions unrelated to fashion."
        )
    }

    # Si se pasa un idioma no reconocido, usamos español como fallback
    system_prompt = system_prompts.get(lang.lower(), system_prompts["es"])
    print(system_prompt)
    # Cargar imágenes como bloques base64 para el mensaje
    image_blocks = []
    for path in image_paths:
        try:
            data_url = get_data_url(path)
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        except Exception as e:
            print(f"⚠️ No se pudo procesar la imagen {path}: {e}")
    
    if not image_blocks:
        raise ValueError("Ninguna de las imágenes es válida.")

    # Construir mensajes para el modelo
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": {
                    "es": "Las siguientes imágenes representan prendas similares encontradas.",
                    "en": "The following images show visually similar garments found in the catalog."
                }.get(lang.lower())},
                *image_blocks
            ]
        }
    ]
    
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
    except Exception as exc:
        raise RuntimeError(f"Error al llamar al API de OpenAI: {exc}")

    reply = (response.choices[0].message.content or "").strip()
    return reply
