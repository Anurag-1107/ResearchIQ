from backend.app.pipelines.embedding import EmbeddingPipeline


def main():
    print("\n=== LOADING EMBEDDING MODEL ===")

    pipeline = EmbeddingPipeline()

    print(f"Model: {pipeline.model_name}")
    print("Model loaded successfully.")

    # Test 1: Single embedding
    print("\n=== SINGLE TEXT EMBEDDING ===")

    text = "Self-attention allows a model to focus on different parts of an input sequence."

    embedding = pipeline.embed_text(text)

    print(f"Input Text: {text}")
    print(f"Embedding Dimension: {len(embedding)}")
    print(f"First 10 Values: {embedding[:10]}")

    assert len(embedding) == 384

    # Test 2: Batch embedding
    print("\n=== BATCH EMBEDDING ===")

    texts = [
        "What is self-attention in a Transformer?",
        "Explain how attention mechanisms work.",
        "How do I bake a chocolate cake?",
    ]

    embeddings = pipeline.embed_texts(texts)

    print(f"Number of Input Texts: {len(texts)}")
    print(f"Number of Embeddings: {len(embeddings)}")

    for index, vector in enumerate(embeddings):
        print(
            f"Text {index + 1} Embedding Dimension: "
            f"{len(vector)}"
        )

        assert len(vector) == 384

    print("\n=== EMBEDDING VALIDATION SUCCESSFUL ===")


if __name__ == "__main__":
    main()