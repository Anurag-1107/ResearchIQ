from backend.app.pipelines.chunking import DocumentChunkingPipeline
from backend.app.pipelines.document_ingestion import DocumentIngestionPipeline
from backend.app.pipelines.embedding import EmbeddingPipeline
from backend.app.pipelines.vector_store import VectorStorePipeline


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

    print("\n=== STEP 3: GENERATING EMBEDDINGS ===")

    embedding_pipeline = EmbeddingPipeline()

    embeddings = embedding_pipeline.embed_texts(
        [chunk.text for chunk in chunks]
    )

    print(f"Generated {len(embeddings)} embeddings.")

    print("\n=== STEP 4: STORING IN CHROMADB ===")

    vector_store = VectorStorePipeline()

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(
        f"Stored chunks in collection: "
        f"{vector_store.get_collection_count()}"
    )

    print("\n=== STEP 5: DENSE RETRIEVAL TEST ===")

    query = "How does self-attention work in the Transformer?"

    print(f"Query: {query}")

    query_embedding = embedding_pipeline.embed_text(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    print("\n=== TOP RETRIEVAL RESULTS ===")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print(f"\n--- Rank {rank} ---")
        print(f"Filename: {metadata['filename']}")
        print(f"Page: {metadata['page_number']}")
        print(f"Chunk Index: {metadata['chunk_index']}")
        print(f"Distance: {distance:.4f}")
        print(f"Text: {document[:500]}")

    print("\n=== END-TO-END DENSE RETRIEVAL SUCCESSFUL ===")


if __name__ == "__main__":
    main()