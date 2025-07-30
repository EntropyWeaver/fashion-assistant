"""
FastAPI routes for the fashion stylist assistant.

This module defines the REST endpoint used by the front‑end to submit
queries. It relies on the retrieval engine to find similar garments from
the local dataset and on the LLM assistant to formulate a natural
language answer. The endpoint accepts an image file and a text prompt
from the client and returns JSON containing the retrieved image paths,
the full prompt sent to the LLM and the final answer.
"""

import os
from typing import List

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.retrieval.engine import RetrievalEngine
from app.llm.assistant import ask_fashion_assistant
from ..llm.assistant import contains_offensive_language



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
        image: UploadFile = File(..., description="Imagen de prenda"),
        text: str = Form(..., description="Consulta de moda"),
        k: int = Form(5, description="Número de similares a recuperar"),
    ) -> JSONResponse:
        print("🟢 [DEBUG] Inicio de /query")  # DEBUG
        
        # Leer imagen
        try:
            contents = await image.read()
            if not contents:
                raise HTTPException(status_code=400, detail="La imagen está vacía")
            print(f"📦 [DEBUG] Imagen recibida: {image.filename} ({len(contents)} bytes)")  # DEBUG
        except Exception as e:
            print("❌ [DEBUG] Error al leer imagen:", e)
            raise

        # Guardar imagen temporal en carpeta 'temp'
        tmp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, image.filename)
        try:
            with open(tmp_path, "wb") as f:
                f.write(contents)
            print(f"💾 [DEBUG] Imagen guardada en: {tmp_path}")  # DEBUG
        except Exception as e:
            print("❌ [DEBUG] Error al guardar imagen:", e)
            raise HTTPException(status_code=500, detail="No se pudo guardar la imagen temporal")

        # FAISS retrieval
        try:
            if contains_offensive_language(text):
                print("🚫 [DEBUG] Solicitud bloqueada por lenguaje ofensivo")
                return JSONResponse(
                {
                    "similar_images": [],
                    "prompt": text,
                    "answer": "Lo siento, no puedo ayudarte con esa solicitud.",
                    "blocked": True
                },
                status_code=200,
            )
                
            print("🔍 [DEBUG] Llamando a retrieval_engine.search()")  # DEBUG
            similar_items = retrieval_engine.search(tmp_path, k=max(1, int(k)))
            print(f"✅ [DEBUG] FAISS devolvió {len(similar_items)} resultados")  # DEBUG
        except Exception as exc:
            print("❌ [DEBUG] Error en FAISS:", exc)
            import traceback; traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error en búsqueda FAISS: {exc}")

        # LLM call
        try:
            print("🤖 [DEBUG] Llamando a ask_fashion_assistant()")  # DEBUG
            answer, full_prompt = ask_fashion_assistant(user_query=text, similar_images=similar_items)
            print("✅ [DEBUG] Respuesta generada por LLM")  # DEBUG
        except Exception as exc:
            print("❌ [DEBUG] Error en LLM:", exc)
            import traceback; traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error en el modelo de lenguaje: {exc}")

        # Limpieza temporal
        try:
            os.remove(tmp_path)
            print("🧹 [DEBUG] Imagen temporal eliminada")  # DEBUG
        except Exception:
            pass

        print("🔚 [DEBUG] Fin de /query, devolviendo JSON")  # DEBUG
        return JSONResponse(
            {
                "similar_images": [item["image"] for item in similar_items],
                "prompt": full_prompt,
                "answer": answer,
            }
        )
    app.include_router(router)
    return app

app = create_app()