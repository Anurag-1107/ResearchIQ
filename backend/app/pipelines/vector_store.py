from pathlib import Path

import chromadb

from backend.app.schemas.document import DocumentChunk


class VectorStorePipeline:
    """
    Stores and retrieves document embeddings using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str | Path = "storage/chroma",
        collection_name: str = "researchiq_documents",
    ) -> None:
        persist_directory = Path(persist_directory)

        persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": (
                    "ResearchIQ document chunk embeddings"
                )
            },
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store document chunks with their embeddings and provenance metadata.
        """

        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match "
                "the number of embeddings."
            )

        ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        documents = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
    self,
    query_embedding: list[float],
    top_k: int = 5,
    document_id: str | None = None,
  ) -> dict:
      """
      Perform dense vector similarity search.

      If document_id is provided, search only within
      that specific document.
      """

      if not query_embedding:
          raise ValueError(
              "Query embedding cannot be empty."
          )

      if top_k <= 0:
          raise ValueError(
              "top_k must be greater than 0."
          )

      query_params = {
          "query_embeddings": [query_embedding],
          "n_results": top_k,
          "include": [
              "documents",
              "metadatas",
              "distances",
          ],
      }

      if document_id is not None:

          if not document_id.strip():
              raise ValueError(
                  "Document ID cannot be empty."
              )

          query_params["where"] = {
              "document_id": document_id
          }

      return self.collection.query(
          **query_params
      )