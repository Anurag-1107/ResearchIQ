from pathlib import Path

from backend.app.pipelines.document_ingestion import (
    DocumentIngestionPipeline,
)
from backend.app.pipelines.chunking import (
    DocumentChunkingPipeline,
)
from backend.app.pipelines.embedding import (
    EmbeddingPipeline,
)
from backend.app.pipelines.vector_store import (
    VectorStorePipeline,
)
from backend.app.pipelines.reranker import Reranker
from backend.app.pipelines.generation_pipeline import (
    GenerationPipeline,
)


def main():
    print("\n=== STEP 1: DOCUMENT INGESTION ===")

    pdf_path = Path("data/Research_paper.pdf")

    ingestion_pipeline = DocumentIngestionPipeline()
    document = ingestion_pipeline.ingest(
        file_path=pdf_path,
    )

    print(
        f"Extracted {document.metadata.page_count} pages."
    )

    print("\n=== STEP 2: DOCUMENT CHUNKING ===")

    chunking_pipeline = DocumentChunkingPipeline()
    chunks = chunking_pipeline.chunk_document(
        document=document,
    )

    print(f"Created {len(chunks)} chunks.")

    print("\n=== STEP 3: INITIALIZING COMPONENTS ===")

    embedding_pipeline = EmbeddingPipeline()

    vector_store = VectorStorePipeline()

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embedding_pipeline.embed_texts(
            [chunk.text for chunk in chunks]
        ),
    )

    reranker = Reranker()

    print("Components initialized successfully.")

    print("\n=== STEP 4: INITIALIZING GENERATION PIPELINE ===")

    generation_pipeline = GenerationPipeline(
        chunks=chunks,
        embedding_pipeline=embedding_pipeline,
        vector_store=vector_store,
        reranker=reranker,
    )

    print("Generation pipeline initialized successfully.")

    print("\n=== STEP 5: RAG GENERATION TEST ===")

    question = "How does scaled dot-product attention work?"

    print(f"Question: {question}")

    result = generation_pipeline.generate_answer(
        question=question,
    )