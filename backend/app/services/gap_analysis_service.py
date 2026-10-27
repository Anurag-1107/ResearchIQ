from backend.app.pipelines.retrieval_pipeline import (
    RetrievalPipeline,
)
from backend.app.pipelines.embedding import EmbeddingPipeline
from backend.app.pipelines.reranker import Reranker
from backend.app.pipelines.vector_store import VectorStorePipeline
from backend.app.prompts.gap_analysis_prompt import (
    SYSTEM_PROMPT,
    build_gap_analysis_prompt,
)
from backend.app.schemas.document import DocumentChunk
from backend.app.services.document_registry import (
    DocumentRegistry,
)
from backend.app.services.llm_service import LLMService


class GapAnalysisService:
    """
    Handles research gap analysis across multiple documents.

    Evidence is retrieved independently from each selected
    document using multiple research-focused retrieval queries.
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
        top_k: int,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieve research-gap-related evidence from one document.

        Multiple targeted queries are used because limitations,
        future work, methodology, and conclusions may appear in
        different parts of a research paper.
        """

        chunks = self.document_registry.get_document(
            document_id
        )

        retrieval_pipeline = RetrievalPipeline(
            chunks=chunks,
            embedding_pipeline=self.embedding_pipeline,
            vector_store=self.vector_store,
            reranker=self.reranker,
        )

        research_queries = [
            "research problem and motivation",
            "limitations of the proposed approach",
            "future work and possible improvements",
            "challenges and weaknesses",
            "experimental results and findings",
            "conclusion and future directions",
        ]

        candidate_evidence: dict[
            str,
            tuple[DocumentChunk, float],
        ] = {}

        for query in research_queries:

            evidence = retrieval_pipeline.retrieve(
                query=query,
                top_k=top_k,
            )

            for chunk, score in evidence:

                existing = candidate_evidence.get(
                    chunk.chunk_id
                )

                if (
                    existing is None
                    or score > existing[1]
                ):
                    candidate_evidence[
                        chunk.chunk_id
                    ] = (chunk, score)

        ranked_evidence = sorted(
            candidate_evidence.values(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked_evidence[:top_k]

    def _build_document_context(
        self,
        evidence: list[tuple[DocumentChunk, float]],
    ) -> str:
        """
        Build context for one document.
        """

        context_parts = []

        for rank, (chunk, score) in enumerate(
            evidence,
            start=1,
        ):

            context_parts.append(
                f"""
[EVIDENCE {rank}]
Filename: {chunk.filename}
Page: {chunk.page_number}
Chunk Index: {chunk.chunk_index}

Content:
{chunk.text}
"""
            )

        return "\n".join(context_parts)

    def analyze(
        self,
        document_ids: list[str],
        top_k: int = 5,
    ) -> dict:
        """
        Analyze multiple research papers and identify
        potential research gaps.
        """

        if len(document_ids) < 2:
            raise ValueError(
                "At least two documents are required "
                "for research gap analysis."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        document_ids = list(
            dict.fromkeys(document_ids)
        )

        if len(document_ids) < 2:
            raise ValueError(
                "At least two different documents are "
                "required for research gap analysis."
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

            evidence = self._retrieve_document_evidence(
                document_id=document_id,
                top_k=top_k,
            )

            if not evidence:
                continue

            context = self._build_document_context(
                evidence
            )

            filename = evidence[0][0].filename

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
                        "reranker_score": score,
                    }
                )

        if len(document_contexts) < 2:
            raise ValueError(
                "Insufficient evidence was found "
                "for research gap analysis."
            )

        user_prompt = build_gap_analysis_prompt(
            document_contexts=document_contexts,
        )

        answer = self.llm_service.generate_answer(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=4000,
        )

        return {
            "answer": answer,
            "sources": sources,
        }