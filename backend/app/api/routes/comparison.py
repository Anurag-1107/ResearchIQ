from fastapi import APIRouter, HTTPException

from backend.app.api.routes.documents import (
    document_registry,
    document_service,
)
from backend.app.pipelines.reranker import Reranker
from backend.app.schemas.query import (
    ComparisonRequest,
    ComparisonResponse,
)
from backend.app.services.comparison_service import (
    ComparisonService,
)


router = APIRouter(
    prefix="/compare",
    tags=["Comparison"],
)


reranker = Reranker()


comparison_service = ComparisonService(
    document_registry=document_registry,
    embedding_pipeline=(
        document_service.embedding_pipeline
    ),
    vector_store=document_service.vector_store,
    reranker=reranker,
)


@router.post(
    "",
    response_model=ComparisonResponse,
)
async def compare_documents(
    request: ComparisonRequest,
) -> ComparisonResponse:
    """
    Compare multiple uploaded research documents.
    """

    # Remove duplicate document IDs while preserving order.
    document_ids = list(
        dict.fromkeys(request.document_ids)
    )

    if len(document_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two different documents "
                "are required for comparison."
            ),
        )

    # Make sure every selected document exists.
    for document_id in document_ids:

        if not document_registry.has_document(
            document_id
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document not found: "
                    f"{document_id}"
                ),
            )

    try:

        result = comparison_service.compare(
            document_ids=document_ids,
            question=request.question,
            top_k=request.top_k,
        )

        return ComparisonResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Comparison processing failed: "
                f"{str(error)}"
            ),
        )