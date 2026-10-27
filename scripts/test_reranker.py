from backend.app.pipelines.chunking import DocumentChunkingPipeline
from backend.app.pipelines.document_ingestion import DocumentIngestionPipeline
from backend.app.pipelines.reranker import Reranker


def main():
    print("\n=== STEP 1: DOCUMENT INGESTION ===")

    ingestion_pipeline = DocumentIngestionPipeline()

    extracted_document = ingestion_pipeline.extract_pdf(
        "data/raw/Research_paper.pdf"
    )

    print(
        f"Extracted "
        f"{extracted_document.metadata.page_count} pages."
    )

    print("\n=== STEP 2: DOCUMENT CHUNKING ===")

    chunking_pipeline = DocumentChunkingPipeline(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = chunking_pipeline.chunk_document(
        extracted_document
    )

    print(f"Created {len(chunks)} chunks.")

    print("\n=== STEP 3: INITIALIZING RERANKER ===")

    reranker = Reranker()

    print("Reranker initialized successfully.")

    print("\n=== STEP 4: SELECTING TEST CANDIDATES ===")

    # We deliberately provide a mix of relevant
    # and less relevant chunks.
    candidate_indices = [2, 7, 13, 14, 25, 40]

    candidates = [
        chunks[index]
        for index in candidate_indices
    ]

    print(
        f"Selected {len(candidates)} test candidates."
    )

    print("\n=== STEP 5: RERANKING TEST ===")

    query = "How does scaled dot-product attention work?"

    print(f"Query: {query}")

    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=5,
    )

    print("\n=== RERANKED RESULTS ===")

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Rank {rank} ---")
        print(f"Page: {chunk.page_number}")
        print(f"Chunk Index: {chunk.chunk_index}")
        print(f"Reranker Score: {score:.4f}")
        print(f"Text: {chunk.text[:500]}")

    print(
        "\n=== RERANKER VALIDATION SUCCESSFUL ==="
    )


if __name__ == "__main__":
    main()