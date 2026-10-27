from pydantic import BaseModel, Field


# ============================================================
# Sources
# ============================================================

class SourceResponse(BaseModel):
    """
    Source metadata returned with an answer.
    """

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    reranker_score: float


# ============================================================
# Single Document Q&A
# ============================================================

class QueryRequest(BaseModel):
    """
    Request schema for asking a question about
    an uploaded document.
    """

    document_id: str = Field(
        ...,
        min_length=1,
        description="ID of the uploaded document.",
    )

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the document.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of evidence chunks to use.",
    )


class VerificationClaim(BaseModel):
    """
    Verification result for an individual claim.
    """

    claim: str
    status: str
    evidence_indexes: list[int]
    reason: str


class VerificationResponse(BaseModel):
    """
    Overall citation verification result.
    """

    claims: list[VerificationClaim]
    overall_status: str


class QueryResponse(BaseModel):
    """
    Response schema for a document question.
    """

    document_id: str
    question: str
    answer: str
    sources: list[SourceResponse]
    verification: VerificationResponse


# ============================================================
# Document Comparison
# ============================================================

class ComparisonRequest(BaseModel):
    """
    Request schema for comparing multiple
    research documents.
    """

    document_ids: list[str] = Field(
        ...,
        min_length=2,
        description=(
            "IDs of the research documents "
            "to compare."
        ),
    )

    question: str = Field(
        ...,
        min_length=1,
        description=(
            "Question or aspect to compare "
            "across the documents."
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "Number of evidence chunks to retrieve "
            "from each document."
        ),
    )


class ComparisonResponse(BaseModel):
    """
    Response schema for document comparison.
    """

    question: str
    answer: str
    sources: list[SourceResponse]


# ============================================================
# Research Gap Analysis
# ============================================================

class GapAnalysisRequest(BaseModel):
    """
    Request schema for research gap analysis
    across multiple documents.
    """

    document_ids: list[str] = Field(
        ...,
        min_length=2,
        description=(
            "IDs of the research documents "
            "to analyze."
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "Number of evidence chunks to retrieve "
            "from each document."
        ),
    )


class GapAnalysisResponse(BaseModel):
    """
    Response schema for research gap analysis.
    """

    answer: str
    sources: list[SourceResponse]