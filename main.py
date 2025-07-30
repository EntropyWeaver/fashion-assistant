"""
Entry point to run the FastAPI application.

This script simply imports the FastAPI app from the API package and runs
it with Uvicorn.  When deploying in production you may wish to use a
ASGI server like Hypercorn or Gunicorn instead of running directly via
this script.
"""

from app.api.main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
