"""
Interface to the OpenAI API for providing fashion advice.

This module contains functions to build a conversational prompt based on
the user's query and the list of similar images returned by the
retrieval engine.  It also implements a simple guardrail to prevent
sending requests containing offensive, sexual or violent language.  If
such language is detected the assistant will respond politely without
calling the LLM API.
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple

import openai

# A minimal list of disallowed keywords.  This list can be extended
# depending on the domain requirements.  Words are matched in a case
# insensitive manner.
BANNED_KEYWORDS = {
    "sexo",
    "sexual",
    "porn",
    "pornografía",
    "violencia",
    "violento",
    "asesinar",
    "matar",
    "insulto",
    "puta",
    "mierda",
}


def contains_offensive_language(text: str) -> bool:
    """Check whether the input text contains any banned keyword.

    The comparison is case insensitive.  This is a simple guardrail; for
    more robust filtering consider using a moderation API.

    Parameters
    ----------
    text : str
        The text to analyse.

    Returns
    -------
    bool
        True if a banned keyword is found, False otherwise.
    """
    lower_text = text.lower()
    for word in BANNED_KEYWORDS:
        if word in lower_text:
            return True
    return False


def _build_similar_images_description(similar_images: List[Dict[str, object]]) -> str:
    """Construct a human‑readable summary of the similar images.

    Each image entry contains a filename, category and distance.  The
    summary enumerates the images by category and distance.  Only the
    category and distance are exposed to the language model to avoid
    leaking file system details.

    Parameters
    ----------
    similar_images : list of dict
        Output of the retrieval engine's search method.

    Returns
    -------
    str
        A string description of the similar images.
    """
    lines = []
    for idx, item in enumerate(similar_images, start=1):
        category = item.get("category", "desconocida")
        distance = item.get("distance", 0.0)
        lines.append(f"{idx}. Categoría: {category}, similitud: {1 - distance:.2f}")
    return "\n".join(lines)


def _build_prompt(user_query: str, similar_images: List[Dict[str, object]]) -> str:
    """Compose the prompt to be sent to the OpenAI API.

    The prompt instructs the model to act as a fashion stylist and
    leverages the information about similar garments to enrich its
    response.  The user query is appended at the end to preserve
    conversational context.

    Parameters
    ----------
    user_query : str
        The question provided by the user.
    similar_images : list of dict
        The list of similar images returned by the retrieval engine.

    Returns
    -------
    str
        The full prompt string.
    """
    description = _build_similar_images_description(similar_images)
    prompt = (
        "Eres un estilista virtual experto en moda y tendencias. "
        "A partir de la siguiente lista de prendas similares a la que envía el usuario, "
        "ofrece consejos de estilo sobre cómo combinar la prenda de la imagen y "
        "otras sugerencias relevantes. Utiliza un tono amistoso, alentador y profesional.\n\n"
        "Prendas similares encontradas:\n"
        f"{description}\n\n"
        "Consulta del usuario: " + user_query.strip()
    )
    return prompt


def ask_fashion_assistant(
    user_query: str, similar_images: List[Dict[str, object]], model: str = "gpt-4o"
) -> Tuple[str, str]:
    """Generate a fashion advice response using OpenAI's chat model.

    If the user's query contains offensive language, the function
    immediately returns a neutral refusal.  Otherwise it composes a
    prompt enriched with descriptions of similar garments, submits it to
    the OpenAI API and returns the assistant's reply.

    Parameters
    ----------
    user_query : str
        The text input from the user describing their fashion concern.
    similar_images : list of dict
        A list of similar image dictionaries as returned by
        :meth:`app.retrieval.engine.RetrievalEngine.search`.
    model : str, default="gpt-4o"
        The name of the OpenAI chat model to use (e.g. "gpt-4" or
        "gpt-4o").

    Returns
    -------
    answer : str
        The assistant's response in Spanish.  If the query is
        inappropriate the answer is a polite refusal.
    full_prompt : str
        The full prompt that was sent to the OpenAI API.  This is
        returned for transparency and debugging purposes.
    """
    # Check for offensive content
    if contains_offensive_language(user_query):
        refusal = "Lo siento, no puedo ayudarte con esa solicitud."
        return refusal, user_query

    # Build prompt with similar images information
    prompt = _build_prompt(user_query, similar_images)

    # Retrieve API key and ensure it's set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La variable de entorno OPENAI_API_KEY no está configurada. Establece tu clave de API de OpenAI para utilizar el asistente."
        )
    openai.api_key = api_key

    # Compose messages for chat model.  A system prompt sets the role and
    # guidelines; the user prompt contains our constructed message.
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un estilista virtual que ofrece consejos de moda personalizados. "
                "Tu objetivo es ayudar al usuario a combinar prendas, sugerir colores y accesorios "
                "y explicar por qué tus recomendaciones funcionan. Responde siempre en español."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Call the OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
    except Exception as exc:
        raise RuntimeError(f"Fallo al llamar al API de OpenAI: {exc}")

    answer = response.choices[0].message.get("content", "").strip()
    return answer, prompt
