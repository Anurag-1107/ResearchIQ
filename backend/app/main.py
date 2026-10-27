from fastapi import FastAPI

from backend.app.api.routes.documents import (
    router as documents_router,
)

from backend.app.api.routes.query import (
    router as query_router,
)

from backend.app.api.routes.comparison import (
    router as comparison_router,
)

from backend.app.api.routes.gap_analysis import (
    router as gap_analysis_router,
)


app = FastAPI(
    title="ResearchIQ API",
    description=(
        "Backend API for ResearchIQ — "
        "An Agentic Multimodal Research Intelligence Platform"
    ),
    version="0.1.0",
)


app.include_router(documents_router)

app.include_router(query_router)

app.include_router(comparison_router)

app.include_router(gap_analysis_router)


@app.get("/")
async def root():

    return {
        "message": "Welcome to ResearchIQ API",
        "status": "running",
    }


@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "service": "ResearchIQ Backend",
        "version": "0.1.0",
    }