import re
import hashlib

from backend.app.schemas.document import (
    DocumentChunk,
    ExtractedDocument,
)


class DocumentChunkingPipeline:
    """
    Converts page-aware extracted documents into retrieval-ready chunks.

    The pipeline preserves page-level provenance and uses natural
    boundaries while ensuring chunks do not start or end in the middle
    of a whitespace-delimited token.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _find_split_position(
        self,
        text: str,
        start: int,
        max_end: int,
    ) -> int:
        """
        Find the best boundary before the maximum chunk size.

        Priority:
        1. Paragraph boundary
        2. Sentence boundary
        3. Line break
        4. Whitespace
        5. Hard boundary
        """

        if max_end >= len(text):
            return len(text)

        search_window = text[start:max_end]

        paragraph_pos = search_window.rfind("\n\n")
        if paragraph_pos > 0:
            return start + paragraph_pos + 2

        sentence_matches = list(
            re.finditer(r"[.!?](?:\s+)", search_window)
        )

        if sentence_matches:
            return start + sentence_matches[-1].end()

        line_break_pos = search_window.rfind("\n")
        if line_break_pos > 0:
            return start + line_break_pos + 1

        whitespace_matches = list(
            re.finditer(r"\s+", search_window)
        )

        if whitespace_matches:
            return start + whitespace_matches[-1].end()

        return max_end

    def _find_overlap_start(
        self,
        text: str,
        end: int,
    ) -> int:
        """
        Find a safe start position for the next chunk.

        The overlap target is approximately chunk_overlap characters,
        but the start is adjusted to a whitespace boundary.
        """

        target_start = max(0, end - self.chunk_overlap)

        if target_start == 0:
            return 0

        # Move backward to the start of the current token.
        while (
            target_start > 0
            and not text[target_start - 1].isspace()
        ):
            target_start -= 1

        # Skip whitespace so the next chunk begins with text.
        while (
            target_start < len(text)
            and text[target_start].isspace()
        ):
            target_start += 1

        return target_start
    
    
    def _generate_chunk_id(
        self,
        filename: str,
        page_number: int,
        chunk_index: int,
        text: str,
        ) -> str:
        """
        Generate a deterministic ID for a chunk.

        The same document content and chunk configuration will produce
        the same chunk ID across repeated ingestion runs.
        """

        content = (
            f"{filename}|"
            f"{page_number}|"
            f"{chunk_index}|"
            f"{text}"
        )

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    def chunk_document(
        self,
        document: ExtractedDocument,
    ) -> list[DocumentChunk]:
        """
        Split each page into overlapping chunks while preserving
        page-level provenance.
        """

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in document.pages:
            page_text = page.text.strip()

            if not page_text:
                continue

            start = 0
            text_length = len(page_text)

            while start < text_length:
                max_end = min(
                    start + self.chunk_size,
                    text_length,
                )

                end = self._find_split_position(
                    text=page_text,
                    start=start,
                    max_end=max_end,
                )

                if end <= start:
                    end = max_end

                chunk_text = page_text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=self._generate_chunk_id(
                                filename=document.metadata.filename,
                                page_number=page.page_number,
                                chunk_index=chunk_index,
                                text=chunk_text,
                            ),
                            document_id=document.metadata.document_id,
                            filename=document.metadata.filename,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            text=chunk_text,
                        )
                    )
                    chunk_index += 1

                if end >= text_length:
                    break

                next_start = self._find_overlap_start(
                    text=page_text,
                    end=end,
                )

                # Guarantee forward progress.
                if next_start <= start:
                    next_start = end

                start = next_start

        return chunks