"""
Interface to the OpenAI API for providing fashion advice.

This module builds a prompt from the user's query plus the list of
similar images, applies a simple guardrail, and calls OpenAI's chat API.
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # carga .env desde la raíz del proyecto

# Palabras vetadas (guardrail simple, case-insensitive)
BANNED_KEYWORDS = {
    "sexo", "sexual", "porn", "pornografía",
    "violencia", "violento", "asesinar", "matar",
    "insulto", "puta", "mierda",
}

def contains_offensive_language(text: str) -> bool:
    lower_text = text.lower()
    return any(word in lower_text for word in BANNED_KEYWORDS)

def _build_similar_images_description(similar_images: List[Dict[str, object]]) -> str:
    """
    Convierte la lista de similares en texto; con FAISS la similitud que usamos es
    coseno.
    """
    lines = []
    for idx, item in enumerate(similar_images, start=1):
        category = item.get("category", "desconocida")
        similarity = float(item.get("similarity", 0.0))
        lines.append(f"{idx}. Categoría: {category}, similitud coseno: {similarity:.4f}")
    return "\n".join(lines)

def _build_prompt(user_query: str, similar_images: List[Dict[str, object]]) -> str:
    description = _build_similar_images_description(similar_images)
    prompt = (
        "Eres un estilista virtual experto en moda y tendencias. "
        "A partir de la siguiente lista de prendas similares a la que envía el usuario, "
        "ofrece consejos de estilo sobre cómo combinar la prenda y sugiere colores, accesorios "
        "y alternativas. Explica brevemente el porqué de tus recomendaciones. Responde SIEMPRE en español.\n\n"
        "Prendas similares encontradas:\n"
        f"{description}\n\n"
        "Consulta del usuario: " + user_query.strip()
    )
    return prompt

def ask_fashion_assistant(
    user_query: str,
    similar_images: List[Dict[str, object]],
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 400,
) -> Tuple[str, str]:
    """
    Devuelve (answer, full_prompt).
    Usa el cliente nuevo de openai>=1.0.0.
    """
    # Guardrail
    if contains_offensive_language(user_query):
        refusal = "Lo siento, no puedo ayudarte con esa solicitud."
        return refusal, user_query

    prompt = _build_prompt(user_query, similar_images)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La variable de entorno OPENAI_API_KEY no está configurada."
        )

    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un estilista virtual que ofrece consejos de moda personalizados. "
                        "Ayuda a combinar prendas, sugerir colores y accesorios, y justifica tus sugerencias. "
                        "Responde siempre en español, con tono amable, claro y útil."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"Fallo al llamar al API de OpenAI: {exc}")

    answer = (resp.choices[0].message.content or "").strip()
    return answer, prompt
