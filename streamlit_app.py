"""
Streamlit front‑end for the fashion stylist assistant.

This application provides a simple interface where users can upload an
image of a garment and enter a question.  The input is sent to the
FastAPI backend to retrieve similar images and generate a fashion advice
response.  The similar images are displayed to the user along with the
assistant's answer.
"""

import os
from io import BytesIO
from typing import List, Dict

import requests
import streamlit as st
from PIL import Image


def call_backend(api_url: str, image_file: BytesIO, filename: str, query: str, k: int = 5) -> Dict[str, object]:
    """Send the uploaded image and query to the backend API.

    Parameters
    ----------
    api_url : str
        The full URL of the backend endpoint (e.g. http://localhost:8000/query).
    image_file : BytesIO
        The file object containing the image data.
    filename : str
        The filename to associate with the uploaded image.
    query : str
        The user's query text.
    k : int, default=5
        Number of similar images to request from the backend.

    Returns
    -------
    dict
        Parsed JSON response from the backend.
    """
    files = {"image": (filename, image_file, "image/jpeg")}
    data = {"text": query, "k": str(k)}
    response = requests.post(api_url, files=files, data=data)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Asistente de Moda", page_icon="🧵")
    st.title("👗 Asistente de Moda")
    st.markdown(
        "Sube una imagen de una prenda y formula tu consulta. El sistema "
        "buscará prendas similares y te ofrecerá consejos de estilo."
    )

    # Inputs
    uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])
    query = st.text_input("¿Cuál es tu consulta de moda?", "¿Con qué puedo combinar esta prenda?")
    k = st.number_input(
        label="Número de recomendaciones", min_value=1, max_value=10, value=5, step=1, help="Cantidad de prendas similares a mostrar"
    )

    if st.button("Buscar y obtener consejo"):
        if uploaded_file is None:
            st.warning("Por favor, sube una imagen antes de continuar.")
        elif not query.strip():
            st.warning("Por favor, introduce una consulta de moda.")
        else:
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000/query")
            with st.spinner("Procesando tu solicitud..."):
                try:
                    # Read file data into memory
                    file_bytes = uploaded_file.getvalue()
                    result = call_backend(
                        api_url=backend_url,
                        image_file=file_bytes,
                        filename=uploaded_file.name,
                        query=query,
                        k=int(k),
                    )
                except Exception as exc:
                    st.error(f"Error al conectar con el backend: {exc}")
                    return

            # Display similar images
            similar_images: List[str] = result.get("similar_images", [])
            if not "blocked" in result:
                st.subheader("Imágenes similares")
                if not similar_images:
                    st.write("No se encontraron imágenes similares.")
            if result.get("blocked"):
                st.warning(f"**{result['answer']}**")
                return
            else:
                for path in similar_images:
                    try:
                        img = Image.open(path)
                        st.image(img, caption=os.path.basename(path), use_column_width=True)
                    except Exception:
                        # If the image can't be loaded, display the path as text
                        st.write(path)

            # Display the LLM answer
            answer = result.get("answer", "")
            st.subheader("Consejo del estilista")
            st.write(answer or "No se pudo generar una respuesta.")


if __name__ == "__main__":
    main()
