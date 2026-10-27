from backend.app.pipelines.chunking import DocumentChunkingPipeline
from backend.app.pipelines.document_ingestion import DocumentIngestionPipeline


def main():
    # Step 1: Extract the PDF page by page
    ingestion_pipeline = DocumentIngestionPipeline()

    extracted_document = ingestion_pipeline.extract_pdf(
        "data/raw/Research_paper.pdf"
    )

    # Step 2: Convert page-aware text into retrieval-ready chunks
    chunking_pipeline = DocumentChunkingPipeline(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = chunking_pipeline.chunk_document(extracted_document)

    # Step 3: Display document information
    print("\n=== DOCUMENT METADATA ===")
    print(f"Document ID: {extracted_document.metadata.document_id}")
    print(f"Filename: {extracted_document.metadata.filename}")
    print(f"Page Count: {extracted_document.metadata.page_count}")

    # Step 4: Display chunking results
    print("\n=== CHUNKING RESULTS ===")
    print(f"Total Chunks: {len(chunks)}")

    # Step 5: Preview the first 5 chunks
    print("\n=== CHUNK PREVIEW ===")

    for chunk in chunks[:5]:
        print(f"\n--- Chunk {chunk.chunk_index} ---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Document ID: {chunk.document_id}")
        print(f"Filename: {chunk.filename}")
        print(f"Source Page: {chunk.page_number}")
        print(f"Text Length: {len(chunk.text)} characters")
        print(f"Starts With: {chunk.text[:80]}")
        print(f"Ends With: {chunk.text[-80:]}")

    # Step 6: Verify provenance
    print("\n=== PROVENANCE CHECK ===")

    for chunk in chunks:
        assert (
            chunk.document_id
            == extracted_document.metadata.document_id
        )
        assert (
            chunk.filename
            == extracted_document.metadata.filename
        )
        assert chunk.page_number >= 1

    print("All chunks passed provenance validation.")
    print("\n=== INGESTION + CHUNKING SUCCESSFUL ===")


if __name__ == "__main__":
    main()