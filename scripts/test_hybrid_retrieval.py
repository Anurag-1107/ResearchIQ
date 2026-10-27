from backend.app.pipelines.chunking import DocumentChunkingPipeline
from backend.app.pipelines.document_ingestion import DocumentIngestionPipeline
from backend.app.pipelines.embedding import EmbeddingPipeline
from backend.app.pipelines.hybrid_retriever import HybridRetriever
from backend.app.pipelines.vector_store import VectorStorePipeline


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

    print("\n=== STEP 3: INITIALIZING COMPONENTS ===")

    embedding_pipeline = EmbeddingPipeline()
    vector_store = VectorStorePipeline()

    hybrid_retriever = HybridRetriever(
        chunks=chunks,
        embedding_pipeline=embedding_pipeline,
        vector_store=vector_store,
    )

    print("Hybrid retriever initialized successfully.")

    print("\n=== STEP 4: HYBRID RETRIEVAL TEST ===")

    query = "How does scaled dot-product attention work?"

    print(f"Query: {query}")

    results = hybrid_retriever.search(
        query=query,
        top_k=5,
        candidate_k=10,
    )

    print("\n=== TOP HYBRID RESULTS ===")

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Rank {rank} ---")
        print(f"Filename: {chunk.filename}")
        print(f"Page: {chunk.page_number}")
        print(f"Chunk Index: {chunk.chunk_index}")
        print(f"RRF Score: {score:.6f}")
        print(f"Text: {chunk.text[:500]}")

    print(
        "\n=== HYBRID RETRIEVAL VALIDATION SUCCESSFUL ==="
    )


if __name__ == "__main__":
    main()