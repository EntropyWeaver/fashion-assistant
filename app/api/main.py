"""
FastAPI routes for the fashion stylist assistant.

This module defines the REST endpoint used by the front‑end to submit
queries. It relies on the retrieval engine to find similar garments from
the local dataset and on the LLM assistant to formulate a natural
language answer. The endpoint accepts an image file and a text prompt
from the client and returns JSON containing the retrieved image paths,
the full prompt sent to the LLM and the final answer.
"""
import traceback
import os
from typing import List

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST

from app.retrieval.engine import RetrievalEngine
from app.services.fashion_advisor import query_fashion_advisor
from app.utils.filters import contains_offensive_language


def create_app() -> FastAPI:
    """Factory function to create a FastAPI app with all routes registered."""
    app = FastAPI(title="Fashion Stylist Assistant")
    router = APIRouter()

    # Initialize the retrieval engine once at startup.  The dataset root can
    # be configured via the DATA_ROOT environment variable or defaults to
    # "/root/data" as described by the user.  Building the index may take
    # some time on the first run but will be reused for subsequent
    # requests.
    data_root = os.getenv("DATA_ROOT", "data")
    try:
        retrieval_engine = RetrievalEngine(data_root=data_root)
    except Exception as exc:
        # It's better to fail fast if the index cannot be built, rather
        # than handle this lazily in the endpoint.  Without a valid index
        # there is no meaningful way to service user queries.
        raise RuntimeError(f"Unable to initialise retrieval engine: {exc}")

    @router.post("/query")
    async def query(
        image: UploadFile = File(..., description="Imagen de prenda que el usuario sube"),
        text: str = Form(..., description="Consulta de moda del usuario"),
        k: int = Form(5, description="Número de imágenes similares a recuperar"),
        lang: str = Form("es", description="Idioma del usuario ('es' o 'en')")
    ) -> JSONResponse:
        """
        Encuentra imágenes similares a una consulta y genera un consejo de moda.

        La ruta acepta una imagen de prenda y una consulta escrita.
        Primero, la imagen se guarda temporalmente y se procesa con el motor
        de recuperación para obtener las `k` imágenes más similares. Luego
        se construye un prompt que incluye la información de dichas
        similitudes y la consulta original del usuario. Finalmente se
        utiliza el modelo de lenguaje para generar una respuesta de moda.
        """
        if contains_offensive_language(text):
            return JSONResponse({"answer": "refusal"})
        # Guardar la imagen recibida en un fichero temporal dentro de /tmp
        try:
            contents = await image.read()
            if not contents:
                raise HTTPException(status_code=400, detail="La imagen está vacía")
        except Exception:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

        tmp_dir = "temp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, image.filename)
        try:
            with open(tmp_path, "wb") as f:
                f.write(contents)
        except Exception:
            raise HTTPException(status_code=500, detail="No se pudo guardar la imagen temporal")

        # Recuperar imágenes similares usando el motor de recuperación
        try:
            similar_items = retrieval_engine.search(tmp_path, k=max(1, int(k)))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error en la búsqueda FAISS: {exc}")

        #Usar todas las imagenes que devuelva FAISS
        image_paths: List[str]
        if similar_items:
            image_paths = [item["image"] for item in similar_items]
        else:
            image_paths = [tmp_path]
        # Enviar la imagen más similar junto con la consulta al modelo GPT‑4o. Si no hay
        # resultados, se devuelve un mensaje indicativo.
        
        try:
            answer = query_fashion_advisor(image_paths, text, lang=lang)
        except Exception as exc:
            print("🧨 Excepción al llamar a query_fashion_advisor():", exc)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Error interno al generar la respuesta.")

        # Limpiar el archivo temporal
        try:
            os.remove(tmp_path)
        except Exception:
            # No hacemos que el fallo en la limpieza interrumpa la petición
            pass

        return JSONResponse(
            {
                "similar_images": [item["image"] for item in similar_items],
                "answer": answer,
            }
        )

    app.include_router(router)
    return app


# Instancia global de FastAPI para uvicorn.  Esto permite ejecutar
# `uvicorn app.api.main:app` directamente.
app = create_app()
