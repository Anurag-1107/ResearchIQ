from sentence_transformers import CrossEncoder

from backend.app.schemas.document import DocumentChunk


class Reranker:
    """
    Reranks retrieved document chunks using a cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name

        print(
            f"Loading reranker model: {self.model_name}"
        )

        self.model = CrossEncoder(
            self.model_name
        )

    def rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Score each query-document pair and return
        the highest-ranked candidates.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not candidates:
            raise ValueError(
                "Candidates cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        pairs = [
            [query, chunk.text]
            for chunk in candidates
        ]

        scores = self.model.predict(
            pairs
        )

        ranked_results = sorted(
            zip(candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            (
                chunk,
                float(score),
            )
            for chunk, score in ranked_results[:top_k]
        ]