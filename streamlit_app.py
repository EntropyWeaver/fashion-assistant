"""
Streamlit front‑end for the fashion stylist assistant.

This application provides a simple interface where users can upload an
image of a garment and enter a question.  The input is sent to the
FastAPI backend to retrieve similar images and generate a fashion advice
response.  The similar images are displayed to the user along with the
assistant's answer.
"""

import os
import base64
from io import BytesIO
from typing import List, Dict

import requests
import streamlit as st
from PIL import Image
import json

def _to_data_url(img: Image.Image) -> str:
    """Return a data URL for the given image."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def thumbnail_html(img: Image.Image, width: int = 150) -> str:
    """Create HTML code for a clickable thumbnail that opens the image."""
    data_url = _to_data_url(img)
    return f"<a href='{data_url}' target='_blank'><img src='{data_url}' width='{width}'/></a>"

if not 'lang' in st.session_state:
        st.session_state["lang"] = 'es'

def call_backend(api_url: str, image_file: BytesIO, filename: str, query: str, k: int = 5, lang: str = 'es') -> Dict[str, object]:
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
    data = {"text": query, "k": str(k), "lang": lang}
    response = requests.post(api_url, files=files, data=data)
    response.raise_for_status()
    return response.json()

    
def load_translations(lang: str) -> dict:
    path = f"locals/{lang}.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(page_title="Asistente de Moda", page_icon="🧵")
    lang = st.selectbox("🌍Idioma / Language", ["es", "en"], index=0)
    t = load_translations(lang)
    st.title(t["title"])
    st.markdown(
        t["description"]
    )
    st.session_state["lang"] = lang

    # Inputs
    uploaded_files = st.file_uploader(
        t["upload_label"], type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded_files:
        st.subheader(t.get("uploaded_images", "Uploaded images"))
        cols = st.columns(len(uploaded_files))
        for col, up in zip(cols, uploaded_files):
            try:
                img = Image.open(up)
                col.markdown(thumbnail_html(img), unsafe_allow_html=True)
                col.caption(up.name)
            except Exception:
                col.write(up.name)

    query = st.text_input(t["query_label"], t["default_query"])
    k = st.number_input(
        label=t["recommendation_label"], min_value=1, max_value=10, value=5, step=1, help="Cantidad de prendas similares a mostrar"
    )

    if st.button(t["button"]):
        if not uploaded_files:
            st.warning(t["warning_image"])
        elif not query.strip():
            st.warning(t["warning_text"])
        else:
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000/query")
            with st.spinner(t["spinner"]):
                try:
                    # Read file data into memory
                    file_bytes = uploaded_files[0].getvalue()
                    result = call_backend(
                        api_url=backend_url,
                        image_file=file_bytes,
                        filename=uploaded_files[0].name,
                        query=query,
                        k=int(k),
                        lang=st.session_state["lang"]
                    )
                except Exception as exc:
                    st.error(f"{t['error_backend']}: {exc}")
                    return

            # Display similar images
            similar_images: List[str] = result.get("similar_images", [])
            st.subheader(t["similar_images"])
            if not similar_images:
                st.write(t["no_similar"])
            else:
                cols = st.columns(len(similar_images))
                for col, path in zip(cols, similar_images):
                    try:
                        img = Image.open(path)
                        col.markdown(thumbnail_html(img), unsafe_allow_html=True)
                        col.caption(os.path.basename(path))
                    except Exception:
                        col.write(path)

            # Display the LLM answer
            answer = result.get("answer", "")
            st.subheader(t["stylist_answer"])
            st.write(answer or t["no_answer"])


if __name__ == "__main__":
    main()
