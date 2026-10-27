from backend.app.schemas.document import DocumentChunk


class DocumentRegistry:
    """
    Stores processed documents and their chunks in memory.

    This is intended for the MVP backend. The data will be
    lost when the server restarts.
    """

    def __init__(self) -> None:
        self._documents: dict[
            str,
            list[DocumentChunk],
        ] = {}

    def add_document(
        self,
        document_id: str,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Store chunks for a processed document.
        """

        if not document_id.strip():
            raise ValueError(
                "Document ID cannot be empty."
            )

        if not chunks:
            raise ValueError(
                "Document chunks cannot be empty."
            )

        self._documents[document_id] = chunks

    def get_document(
        self,
        document_id: str,
    ) -> list[DocumentChunk]:
        """
        Retrieve chunks for a document.
        """

        if document_id not in self._documents:
            raise KeyError(
                f"Document not found: {document_id}"
            )

        return self._documents[document_id]

    def has_document(
        self,
        document_id: str,
    ) -> bool:
        """
        Check whether a document exists in the registry.
        """

        return document_id in self._documents
    
    def list_documents(self) -> list[str]:
        """
        Return the IDs of all registered documents.
        """

        return list(
            self._documents.keys()
        )