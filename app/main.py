from fastapi import FastAPI
from app.routers import extraction

app = FastAPI(title="Intelligent Document Understanding API")

app.include_router(extraction.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}