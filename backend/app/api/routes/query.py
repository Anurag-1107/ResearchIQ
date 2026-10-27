from fastapi import APIRouter, HTTPException

from backend.app.api.routes.documents import (
    document_registry,
    document_service,
)
from backend.app.pipelines.generation_pipeline import (
    GenerationPipeline,
)
from backend.app.pipelines.reranker import Reranker
from backend.app.schemas.query import (
    QueryRequest,
    QueryResponse,
)


router = APIRouter(
    prefix="/query",
    tags=["Query"],
)


reranker = Reranker()


@router.post(
    "",
    response_model=QueryResponse,
)
async def query_document(
    request: QueryRequest,
) -> QueryResponse:
    """
    Ask a question about an uploaded document.
    """

    if not document_registry.has_document(
        request.document_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:
        chunks = document_registry.get_document(
            request.document_id
        )

        generation_pipeline = GenerationPipeline(
            chunks=chunks,
            embedding_pipeline=(
                document_service.embedding_pipeline
            ),
            vector_store=document_service.vector_store,
            reranker=reranker,
        )

        result = generation_pipeline.generate_answer(
            question=request.question,
            top_k=request.top_k,
        )

        return QueryResponse(
            document_id=request.document_id,
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            verification=result["verification"],
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(error)}",
        )