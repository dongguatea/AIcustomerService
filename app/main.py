from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.actions import router as actions_router
from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.graph.runtime import close_graph, init_graph

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_graph()
    yield
    await close_graph()


app = FastAPI(title="MewHelp", version="0.1.0", lifespan=lifespan)
app.include_router(actions_router)
app.include_router(chat_router)
app.include_router(extract_router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
