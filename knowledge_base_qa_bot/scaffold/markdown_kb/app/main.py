from fastapi import FastAPI
from pathlib import Path
from dotenv import load_dotenv

from .indexer import load_index_json, build_index
from .routes import router

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

app = FastAPI(title="Markdown Knowledge Base Q&A Bot")
app.include_router(router)
load_dotenv(PROJECT_DIR / ".env")

@app.on_event("startup")
def load_persisted_index():
      _, sections_count = load_index_json()
      if sections_count == 0:
          build_index()
