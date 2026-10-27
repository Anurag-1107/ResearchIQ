from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """
    Represents extracted text from a single PDF page.
    """

    page_number: int = Field(
        ...,
        ge=1,
        description="One-based page number in the source PDF",
    )

    text: str = Field(
        ...,
        description="Extracted text content from the page",
    )


class DocumentMetadata(BaseModel):
    """
    Represents metadata associated with an ingested document.
    """

    document_id: str
    filename: str
    page_count: int


class ExtractedDocument(BaseModel):
    """
    Represents a PDF after page-aware text extraction.
    """

    metadata: DocumentMetadata
    pages: list[PageContent]
    
class DocumentChunk(BaseModel):
    """
    Represents a retrieval-ready chunk with page-level provenance.
    """

    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(
        ...,
        ge=1,
        description="Source page number in the original PDF",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based chunk index within the document",
    )
    text: str