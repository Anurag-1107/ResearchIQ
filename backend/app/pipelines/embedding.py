from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:
    """
    Converts text into dense vector embeddings using a
    pretrained BGE embedding model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a single piece of text.
        """

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a batch.
        """

        if not texts:
            raise ValueError("Texts cannot be empty.")

        if any(not text or not text.strip() for text in texts):
            raise ValueError(
                "All texts must contain non-empty content."
            )

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()