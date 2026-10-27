from collections import defaultdict

from backend.app.pipelines.bm25_retriever import BM25Retriever
from backend.app.pipelines.embedding import EmbeddingPipeline
from backend.app.pipelines.vector_store import VectorStorePipeline
from backend.app.schemas.document import DocumentChunk


class HybridRetriever:
    """
    Combines dense retrieval and BM25 retrieval
    using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_pipeline: EmbeddingPipeline,
        vector_store: VectorStorePipeline,
        rrf_k: int = 60,
    ) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        self.chunks = chunks
        self.embedding_pipeline = embedding_pipeline
        self.vector_store = vector_store
        self.bm25_retriever = BM25Retriever(chunks)
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieve candidates using dense and BM25 search,
        then combine their rankings using RRF.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

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

        # All chunks passed to this HybridRetriever belong
        # to the same document.
        document_id = self.chunks[0].document_id

        dense_results = self._dense_search(
            query=query,
            candidate_k=candidate_k,
            document_id=document_id,
        )

        # BM25 is already restricted to self.chunks,
        # which belong to the current document.
        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        fused_scores = defaultdict(float)
        chunk_lookup = {}

        self._add_rrf_scores(
            results=dense_results,
            fused_scores=fused_scores,
            chunk_lookup=chunk_lookup,
        )

        self._add_rrf_scores(
            results=bm25_results,
            fused_scores=fused_scores,
            chunk_lookup=chunk_lookup,
        )

        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )[:top_k]

        return [
            (
                chunk_lookup[chunk_id],
                fused_scores[chunk_id],
            )
            for chunk_id in ranked_chunk_ids
        ]

    def _dense_search(
        self,
        query: str,
        candidate_k: int,
        document_id: str,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Perform dense retrieval within a specific document
        and reconstruct DocumentChunk objects from ChromaDB
        results.
        """

        query_embedding = (
            self.embedding_pipeline.embed_text(query)
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            document_id=document_id,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        dense_results = []

        for document, metadata in zip(
            documents,
            metadatas,
        ):
            chunk = next(
                (
                    chunk
                    for chunk in self.chunks
                    if (
                        chunk.text == document
                        and chunk.document_id
                        == metadata["document_id"]
                        and chunk.page_number
                        == metadata["page_number"]
                        and chunk.chunk_index
                        == metadata["chunk_index"]
                    )
                ),
                None,
            )

            if chunk is not None:
                dense_results.append(
                    (chunk, 0.0)
                )

        return dense_results

    def _add_rrf_scores(
        self,
        results: list[tuple[DocumentChunk, float]],
        fused_scores: defaultdict,
        chunk_lookup: dict[str, DocumentChunk],
    ) -> None:
        """
        Add Reciprocal Rank Fusion scores.

        RRF score = 1 / (rrf_k + rank)
        """

        for rank, (chunk, _) in enumerate(
            results,
            start=1,
        ):
            fused_scores[chunk.chunk_id] += (
                1 / (self.rrf_k + rank)
            )

            chunk_lookup[
                chunk.chunk_id
            ] = chunk