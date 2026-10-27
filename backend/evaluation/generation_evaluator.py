from pathlib import Path
import time
from backend.app.pipelines.generation_pipeline import (
    GenerationPipeline,
)
from backend.app.pipelines.reranker import Reranker
from backend.app.services.citation_verification_service import (
    CitationVerificationService,
)
from backend.app.services.document_registry import (
    DocumentRegistry,
)
from backend.app.services.document_service import (
    DocumentService,
)

from backend.evaluation.evaluation_dataset import (
    EVALUATION_DATASET,
)


INSUFFICIENT_EVIDENCE_PHRASE = (
    "I don't have enough information in the provided "
    "research documents to answer this question."
)


class GenerationEvaluator:
    """
    Evaluates ResearchIQ answer generation and grounding.

    The evaluator:
        1. Retrieves evidence using the existing RAG pipeline.
        2. Generates an answer using the existing LLM service.
        3. Verifies the generated answer against the retrieved
           evidence.
        4. Separately tracks legitimate insufficient-evidence
           refusals.
    """

    def __init__(
        self,
        upload_dir: str = "data/uploads",
    ) -> None:

        self.document_registry = (
            DocumentRegistry()
        )

        self.document_service = (
            DocumentService(
                document_registry=(
                    self.document_registry
                ),
            )
        )

        self.reranker = Reranker()

        self.citation_verifier = (
            CitationVerificationService()
        )

        self._load_documents(
            upload_dir=upload_dir,
        )

    def _load_documents(
        self,
        upload_dir: str,
    ) -> None:
        """
        Process all documents required by the
        evaluation dataset.
        """

        upload_path = Path(
            upload_dir
        )

        filenames = {
            item["filename"]
            for item in EVALUATION_DATASET
        }

        for filename in filenames:

            file_path = (
                upload_path / filename
            )

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Evaluation document not found: "
                    f"{file_path}"
                )

            print(
                f"\nProcessing evaluation document: "
                f"{filename}"
            )

            self.document_service.process_document(
                file_path=file_path,
            )

    def _get_document_id(
        self,
        filename: str,
    ) -> str:
        """
        Find a document ID using its filename.
        """

        for document_id in (
            self.document_registry.list_documents()
        ):

            chunks = (
                self.document_registry.get_document(
                    document_id
                )
            )

            if chunks and (
                chunks[0].filename == filename
            ):
                return document_id

        raise ValueError(
            f"Document not found: {filename}"
        )

    def _is_insufficient_evidence_answer(
        self,
        answer: str,
    ) -> bool:
        """
        Determine whether the generated answer is an
        explicit insufficient-evidence refusal.
        """

        normalized_answer = (
            answer.strip()
            .lower()
        )

        normalized_phrase = (
            INSUFFICIENT_EVIDENCE_PHRASE
            .strip()
            .lower()
        )

        return normalized_answer == (
            normalized_phrase
        )

    def _build_evidence(
        self,
        sources: list[dict],
        chunks,
    ) -> list[dict]:
        """
        Convert retrieved source metadata into the
        evidence format required by the verifier.
        """

        evidence = []

        for source in sources:

            matching_chunks = [
                chunk
                for chunk in chunks
                if (
                    chunk.chunk_index
                    == source["chunk_index"]
                )
            ]

            if not matching_chunks:
                continue

            chunk = matching_chunks[0]

            evidence.append(
                {
                    "filename": (
                        chunk.filename
                    ),
                    "page_number": (
                        chunk.page_number
                    ),
                    "chunk_index": (
                        chunk.chunk_index
                    ),
                    "text": chunk.text,
                }
            )

        return evidence

    def evaluate(self) -> dict:
        """
        Evaluate generated answers and their grounding.
        """

        results = []

        supported_claims = 0
        partially_supported_claims = 0
        unsupported_claims = 0

        refusal_count = 0

        overall_status_counts = {
            "SUPPORTED": 0,
            "PARTIALLY_SUPPORTED": 0,
            "UNSUPPORTED": 0,
        }

        for item in EVALUATION_DATASET:

            question = item["question"]
            filename = item["filename"]

            document_id = (
                self._get_document_id(
                    filename
                )
            )

            chunks = (
                self.document_registry.get_document(
                    document_id
                )
            )

            generation_pipeline = (
                GenerationPipeline(
                    chunks=chunks,
                    embedding_pipeline=(
                        self.document_service
                        .embedding_pipeline
                    ),
                    vector_store=(
                        self.document_service
                        .vector_store
                    ),
                    reranker=self.reranker,
                )
            )

            generated_result = (
                generation_pipeline.generate_answer(
                    question=question,
                    top_k=5,
                )
            )

            answer = generated_result[
                "answer"
            ]

            sources = generated_result[
                "sources"
            ]

            evidence = self._build_evidence(
                sources=sources,
                chunks=chunks,
            )

            # ------------------------------------------------
            # Handle legitimate refusal separately.
            # ------------------------------------------------

            if self._is_insufficient_evidence_answer(
                answer
            ):

                refusal_count += 1

                results.append(
                    {
                        "question": question,
                        "filename": filename,
                        "answer": answer,
                        "evidence_refusal": True,
                        "verification": None,
                    }
                )

                continue

            # ------------------------------------------------
            # Verify factual answer.
            # ------------------------------------------------

            time.sleep(2)
            verification = (
                self.citation_verifier.verify(
                    answer=answer,
                    evidence=evidence,
                )
            )

            claims = verification[
                "claims"
            ]

            for claim in claims:

                status = claim["status"]

                if status == "SUPPORTED":

                    supported_claims += 1

                elif (
                    status
                    == "PARTIALLY_SUPPORTED"
                ):

                    partially_supported_claims += 1

                elif (
                    status
                    == "UNSUPPORTED"
                ):

                    unsupported_claims += 1

            overall_status = (
                verification[
                    "overall_status"
                ]
            )

            overall_status_counts[
                overall_status
            ] += 1

            results.append(
                {
                    "question": question,
                    "filename": filename,
                    "answer": answer,
                    "evidence_refusal": False,
                    "verification": verification,
                }
            )

        total_claims = (
            supported_claims
            + partially_supported_claims
            + unsupported_claims
        )

        if total_claims == 0:

            supported_claim_rate = 0.0
            partially_supported_claim_rate = 0.0
            unsupported_claim_rate = 0.0

        else:

            supported_claim_rate = (
                supported_claims
                / total_claims
            )

            partially_supported_claim_rate = (
                partially_supported_claims
                / total_claims
            )

            unsupported_claim_rate = (
                unsupported_claims
                / total_claims
            )

        return {
            "num_examples": len(
                EVALUATION_DATASET
            ),
            "total_claims": total_claims,
            "supported_claims": (
                supported_claims
            ),
            "partially_supported_claims": (
                partially_supported_claims
            ),
            "unsupported_claims": (
                unsupported_claims
            ),
            "supported_claim_rate": (
                supported_claim_rate
            ),
            "partially_supported_claim_rate": (
                partially_supported_claim_rate
            ),
            "unsupported_claim_rate": (
                unsupported_claim_rate
            ),
            "evidence_refusal_count": (
                refusal_count
            ),
            "overall_status_counts": (
                overall_status_counts
            ),
            "results": results,
        }