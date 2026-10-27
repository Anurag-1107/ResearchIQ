from backend.app.pipelines.bm25_retriever import BM25Retriever
from backend.app.pipelines.chunking import DocumentChunkingPipeline
from backend.app.pipelines.document_ingestion import DocumentIngestionPipeline


def main():
    print("\n=== STEP 1: DOCUMENT INGESTION ===")

    ingestion_pipeline = DocumentIngestionPipeline()

    extracted_document = ingestion_pipeline.extract_pdf(
        "data/raw/Research_paper.pdf"
    )

    print(
        f"Extracted {extracted_document.metadata.page_count} pages."
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

    print("\n=== STEP 3: BUILDING BM25 INDEX ===")

    bm25_retriever = BM25Retriever(chunks)

    print("BM25 index created successfully.")

    print("\n=== STEP 4: BM25 RETRIEVAL TEST ===")

    query = "scaled dot-product attention"

    print(f"Query: {query}")

    results = bm25_retriever.search(
        query=query,
        top_k=5,
    )

    print("\n=== TOP BM25 RESULTS ===")

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Rank {rank} ---")
        print(f"Filename: {chunk.filename}")
        print(f"Page: {chunk.page_number}")
        print(f"Chunk Index: {chunk.chunk_index}")
        print(f"BM25 Score: {score:.4f}")
        print(f"Text: {chunk.text[:500]}")

    print("\n=== BM25 RETRIEVAL VALIDATION SUCCESSFUL ===")


if __name__ == "__main__":
    main()