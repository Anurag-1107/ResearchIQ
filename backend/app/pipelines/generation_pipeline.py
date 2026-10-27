from backend.app.pipelines.embedding import EmbeddingPipeline
from backend.app.pipelines.retrieval_pipeline import RetrievalPipeline
from backend.app.pipelines.reranker import Reranker
from backend.app.pipelines.vector_store import VectorStorePipeline
from backend.app.prompts.rag_prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from backend.app.schemas.document import DocumentChunk
from backend.app.services.citation_verification_service import (
    CitationVerificationService,
)
from backend.app.services.llm_service import LLMService


class GenerationPipeline:
    """
    Orchestrates retrieval, context construction,
    prompt creation, LLM answer generation,
    and citation verification.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_pipeline: EmbeddingPipeline,
        vector_store: VectorStorePipeline,
        reranker: Reranker,
    ) -> None:

        self.retrieval_pipeline = RetrievalPipeline(
            chunks=chunks,
            embedding_pipeline=embedding_pipeline,
            vector_store=vector_store,
            reranker=reranker,
        )

        self.llm_service = LLMService()

        self.citation_verification_service = (
            CitationVerificationService(
                llm_service=self.llm_service,
            )
        )

    def _build_context(
        self,
        evidence: list[tuple[DocumentChunk, float]],
    ) -> str:

        context_parts = []

        for rank, (chunk, score) in enumerate(
            evidence,
            start=1,
        ):

            context_parts.append(
                f"""[EVIDENCE {rank}]
Filename: {chunk.filename}
Page: {chunk.page_number}
Chunk Index: {chunk.chunk_index}
Reranker Score: {score:.4f}

Content:
{chunk.text}
"""
            )

        return "\n".join(context_parts)

    def _build_verification_evidence(
        self,
        evidence: list[tuple[DocumentChunk, float]],
    ) -> list[dict]:

        return [
            {
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
            for chunk, _ in evidence
        ]

    def generate_answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:

        # ----------------------------------------------------
        # 1. Retrieve and rerank evidence
        # ----------------------------------------------------

        evidence = self.retrieval_pipeline.retrieve(
            query=question,
            top_k=top_k,
        )

        if not evidence:
            return {
                "question": question,
                "answer": (
                    "I could not find sufficient evidence "
                    "in the document to answer this question."
                ),
                "sources": [],
                "verification": {
                    "claims": [],
                    "overall_status": "UNSUPPORTED",
                },
            }

        # ----------------------------------------------------
        # 2. Build context
        # ----------------------------------------------------

        context = self._build_context(
            evidence
        )

        # ----------------------------------------------------
        # 3. Build RAG prompt
        # ----------------------------------------------------

        user_prompt = build_user_prompt(
            question=question,
            context=context,
        )

        # ----------------------------------------------------
        # 4. Generate initial answer
        # ----------------------------------------------------

        answer = self.llm_service.generate_answer(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # ----------------------------------------------------
        # 5. Prepare evidence for verification
        # ----------------------------------------------------

        verification_evidence = (
            self._build_verification_evidence(
                evidence
            )
        )

        # ----------------------------------------------------
        # 6. Verify generated answer
        # ----------------------------------------------------

        try:

            verification = (
                self.citation_verification_service.verify(
                    answer=answer,
                    evidence=verification_evidence,
                )
            )

        except Exception:

            # Do not break the existing Q&A pipeline if
            # verification fails. The generated answer
            # and retrieved sources remain available.

            verification = {
                "claims": [],
                "overall_status": "VERIFICATION_FAILED",
            }

        # ----------------------------------------------------
        # 7. Build source metadata
        # ----------------------------------------------------

        sources = [
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "reranker_score": score,
            }
            for chunk, score in evidence
        ]

        # ----------------------------------------------------
        # 8. Return final result
        # ----------------------------------------------------

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "verification": verification,
        }