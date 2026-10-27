from backend.app.pipelines.retrieval_pipeline import (
    RetrievalPipeline,
)
from backend.app.pipelines.embedding import EmbeddingPipeline
from backend.app.pipelines.reranker import Reranker
from backend.app.pipelines.vector_store import VectorStorePipeline
from backend.app.prompts.comparison_prompt import (
    SYSTEM_PROMPT,
    build_comparison_prompt,
)
from backend.app.schemas.document import DocumentChunk
from backend.app.services.document_registry import (
    DocumentRegistry,
)
from backend.app.services.llm_service import LLMService


class ComparisonService:
    """
    Handles cross-document comparison.

    Retrieves relevant evidence independently from each
    selected document and sends the separated evidence
    to the LLM for comparison.
    """

    def __init__(
        self,
        document_registry: DocumentRegistry,
        embedding_pipeline: EmbeddingPipeline,
        vector_store: VectorStorePipeline,
        reranker: Reranker,
    ) -> None:

        self.document_registry = document_registry
        self.embedding_pipeline = embedding_pipeline
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm_service = LLMService()

    def _retrieve_document_evidence(
        self,
        document_id: str,
        question: str,
        top_k: int,
    ) -> list[tuple[DocumentChunk, float]]:

        chunks = self.document_registry.get_document(
            document_id
        )

        retrieval_pipeline = RetrievalPipeline(
            chunks=chunks,
            embedding_pipeline=self.embedding_pipeline,
            vector_store=self.vector_store,
            reranker=self.reranker,
        )

        return retrieval_pipeline.retrieve(
            query=question,
            top_k=top_k,
        )

    def _build_document_context(
        self,
        evidence: list[tuple[DocumentChunk, float]],
    ) -> str:

        context_parts = []

        for rank, (chunk, score) in enumerate(
            evidence,
            start=1,
        ):

            context_parts.append(
                f"""
[EVIDENCE {rank}]
Page: {chunk.page_number}
Chunk Index: {chunk.chunk_index}

Content:
{chunk.text}
"""
            )

        return "\n".join(
            context_parts
        )

    def compare(
        self,
        document_ids: list[str],
        question: str,
        top_k: int = 5,
    ) -> dict:
        """
        Compare multiple documents using independently
        retrieved evidence.
        """

        if len(document_ids) < 2:
            raise ValueError(
                "At least two documents are required "
                "for comparison."
            )

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        document_contexts = []
        sources = []

        for document_id in document_ids:

            if not self.document_registry.has_document(
                document_id
            ):
                raise KeyError(
                    f"Document not found: {document_id}"
                )

            evidence = (
                self._retrieve_document_evidence(
                    document_id=document_id,
                    question=question,
                    top_k=top_k,
                )
            )

            if not evidence:
                continue

            filename = evidence[
                0
            ][0].filename

            context = (
                self._build_document_context(
                    evidence
                )
            )

            document_contexts.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "context": context,
                }
            )

            for chunk, score in evidence:

                sources.append(
                    {
                        "document_id": document_id,
                        "filename": chunk.filename,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "reranker_score": float(
                            score
                        ),
                    }
                )

        if len(document_contexts) < 2:
            raise ValueError(
                "Insufficient evidence was found "
                "for comparison."
            )

        user_prompt = (
            build_comparison_prompt(
                question=question,
                document_contexts=document_contexts,
            )
        )

        answer = (
            self.llm_service.generate_answer(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1500,
            )
        )

        if not answer or not answer.strip():
            raise ValueError(
                "Comparison generation returned "
                "an empty answer."
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }