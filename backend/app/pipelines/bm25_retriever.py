import re

from rank_bm25 import BM25Okapi

from backend.app.schemas.document import DocumentChunk


class BM25Retriever:
    """
    Performs keyword-based retrieval over document chunks.
    """

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        self.chunks = chunks

        self.tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_chunks
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into normalized tokens for BM25.
        """

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Return the top BM25 matching chunks and scores.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            (
                self.chunks[index],
                float(scores[index]),
            )
            for index in ranked_indices
        ]