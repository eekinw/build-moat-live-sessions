from fastapi import FastAPI
from pathlib import Path
from dotenv import load_dotenv

from .indexer import load_vector_index, build_index
from .routes import router

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

app = FastAPI(title="Vector RAG Knowledge Base Q&A Bot")
app.include_router(router)


@app.on_event("startup")
def load_persisted_index():
    try:
        _, chunks_count = load_vector_index()
        if chunks_count == 0:
              build_index()
    except Exception as exc:
          print(f"[vector_rag] Rebuilding FAISS index: {exc}", flush=True)
          build_index()
