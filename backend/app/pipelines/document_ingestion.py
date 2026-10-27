from pathlib import Path
from uuid import uuid4

import pymupdf

from backend.app.schemas.document import (
    DocumentMetadata,
    ExtractedDocument,
    PageContent,
)


class DocumentIngestionPipeline:
    """
    Handles page-aware text extraction from PDF documents.
    """

    def extract_pdf(self, file_path: str | Path) -> ExtractedDocument:
        """
        Extract text from a PDF while preserving page-level provenance.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        document = pymupdf.open(file_path)

        try:
            pages: list[PageContent] = []

            for page_index, page in enumerate(document):
                text = page.get_text("text").strip()

                pages.append(
                    PageContent(
                        page_number=page_index + 1,
                        text=text,
                    )
                )

            metadata = DocumentMetadata(
                document_id=str(uuid4()),
                filename=file_path.name,
                page_count=len(document),
            )

            return ExtractedDocument(
                metadata=metadata,
                pages=pages,
            )

        finally:
            document.close()