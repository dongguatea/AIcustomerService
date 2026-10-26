from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="MewHelp", version="0.1.0")
app.include_router(chat_router)
app.include_router(extract_router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
