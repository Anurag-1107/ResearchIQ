from backend.app.pipelines.embedding import EmbeddingPipeline
from backend.app.pipelines.hybrid_retriever import HybridRetriever
from backend.app.pipelines.reranker import Reranker
from backend.app.pipelines.vector_store import VectorStorePipeline
from backend.app.schemas.document import DocumentChunk


class RetrievalPipeline:
    """
    Orchestrates hybrid retrieval followed by
    cross-encoder reranking.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_pipeline: EmbeddingPipeline,
        vector_store: VectorStorePipeline,
        reranker: Reranker,
    ) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        self.hybrid_retriever = HybridRetriever(
            chunks=chunks,
            embedding_pipeline=embedding_pipeline,
            vector_store=vector_store,
        )

        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieve candidates using hybrid retrieval,
        then rerank them using a cross-encoder.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0."
            )

        if top_k > candidate_k:
            raise ValueError(
                "top_k cannot be greater than candidate_k."
            )

        hybrid_results = self.hybrid_retriever.search(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        candidates = [
            chunk
            for chunk, _ in hybrid_results
        ]

        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )