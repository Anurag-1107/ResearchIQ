from pathlib import Path

from backend.app.pipelines.chunking import (
    DocumentChunkingPipeline,
)
from backend.app.pipelines.document_ingestion import (
    DocumentIngestionPipeline,
)
from backend.app.pipelines.embedding import (
    EmbeddingPipeline,
)
from backend.app.pipelines.vector_store import (
    VectorStorePipeline,
)
from backend.app.services.document_registry import (
    DocumentRegistry,
)


class DocumentService:
    """
    Handles the complete document processing workflow:
    ingestion, chunking, embedding, vector storage,
    and document registration.
    """

    def __init__(
        self,
        document_registry: DocumentRegistry,
    ) -> None:
        self.ingestion_pipeline = DocumentIngestionPipeline()

        self.chunking_pipeline = DocumentChunkingPipeline()

        self.embedding_pipeline = EmbeddingPipeline()

        self.vector_store = VectorStorePipeline()

        self.document_registry = document_registry

    def process_document(
        self,
        file_path: str | Path,
    ) -> dict:
        """
        Process a PDF document through the complete pipeline.

        Steps:
        1. Extract text from the PDF.
        2. Split the document into chunks.
        3. Generate embeddings for the chunks.
        4. Store chunks and embeddings in ChromaDB.
        5. Store chunks in the in-memory document registry.
        """

        document = self.ingestion_pipeline.extract_pdf(
            file_path
        )

        chunks = self.chunking_pipeline.chunk_document(
            document
        )

        embeddings = self.embedding_pipeline.embed_texts(
            [chunk.text for chunk in chunks]
        )

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        self.document_registry.add_document(
            document_id=document.metadata.document_id,
            chunks=chunks,
        )

        return {
            "document_id": document.metadata.document_id,
            "filename": document.metadata.filename,
            "page_count": document.metadata.page_count,
            "chunk_count": len(chunks),
        }