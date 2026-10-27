from backend.app.pipelines.document_ingestion import DocumentIngestionPipeline
from backend.app.pipelines.chunking import DocumentChunkingPipeline
from backend.app.pipelines.embedding import EmbeddingPipeline
from backend.app.pipelines.vector_store import VectorStorePipeline
from backend.app.pipelines.reranker import Reranker
from backend.app.pipelines.generation_pipeline import GenerationPipeline


def main() -> None:
    print("\n=== STEP 1: INITIALIZING PIPELINES ===")

    ingestion_pipeline = DocumentIngestionPipeline()
    chunking_pipeline = DocumentChunkingPipeline()
    embedding_pipeline = EmbeddingPipeline()
    vector_store = VectorStorePipeline()
    reranker = Reranker()

    print("Initial pipelines initialized successfully.")

    print("\n=== STEP 2: DOCUMENT INGESTION ===")

    document = ingestion_pipeline.extract_pdf(
        "data/raw/Research_paper.pdf"
    )

    print(
        f"Document ingested successfully: "
        f"{document.metadata.filename}"
    )

    print(
        f"Total pages: {len(document.pages)}"
    )

    print("\n=== STEP 3: DOCUMENT CHUNKING ===")

    chunks = chunking_pipeline.chunk_document(
        document
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("\n=== STEP 4: GENERATING EMBEDDINGS ===")

    embeddings = embedding_pipeline.embed_texts(
        [chunk.text for chunk in chunks]
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    print("\n=== STEP 5: STORING IN VECTOR DATABASE ===")

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("Chunks stored successfully in ChromaDB.")

    print("\n=== STEP 6: INITIALIZING GENERATION PIPELINE ===")

    generation_pipeline = GenerationPipeline(
        chunks=chunks,
        embedding_pipeline=embedding_pipeline,
        vector_store=vector_store,
        reranker=reranker,
    )

    print("Generation pipeline initialized successfully.")

    print("\n=== STEP 7: ASKING QUESTION ===")

    question = "How does scaled dot-product attention work?"

    print(f"Question: {question}")

    print("\n=== STEP 8: RETRIEVING EVIDENCE AND GENERATING ANSWER ===")

    result = generation_pipeline.generate_answer(
        question=question,
        top_k=5,
    )

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES ===")

    for source in result["sources"]:
        print(
            f"Filename: {source['filename']} | "
            f"Page: {source['page_number']} | "
            f"Chunk: {source['chunk_index']} | "
            f"Score: {source['reranker_score']:.4f}"
        )


if __name__ == "__main__":
    main()