from fastapi import APIRouter, HTTPException

from backend.app.api.routes.documents import (
    document_registry,
    document_service,
)
from backend.app.pipelines.reranker import Reranker
from backend.app.schemas.query import (
    GapAnalysisRequest,
    GapAnalysisResponse,
)
from backend.app.services.gap_analysis_service import (
    GapAnalysisService,
)


router = APIRouter(
    prefix="/gap-analysis",
    tags=["Research Gap Analysis"],
)


reranker = Reranker()


gap_analysis_service = GapAnalysisService(
    document_registry=document_registry,
    embedding_pipeline=(
        document_service.embedding_pipeline
    ),
    vector_store=document_service.vector_store,
    reranker=reranker,
)


@router.post(
    "",
    response_model=GapAnalysisResponse,
)
async def analyze_research_gaps(
    request: GapAnalysisRequest,
) -> GapAnalysisResponse:
    """
    Identify potential research gaps across
    multiple uploaded research papers.
    """

    # Remove duplicate document IDs
    # while preserving their original order.
    document_ids = list(
        dict.fromkeys(request.document_ids)
    )

    if len(document_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two different documents "
                "are required for gap analysis."
            ),
        )

    # Verify that every document exists.
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

        result = gap_analysis_service.analyze(
            document_ids=document_ids,
            top_k=request.top_k,
        )

        return GapAnalysisResponse(
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
                "Gap analysis processing failed: "
                f"{str(error)}"
            ),
        )