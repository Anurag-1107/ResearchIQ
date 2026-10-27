from pathlib import Path

from backend.app.pipelines.embedding import EmbeddingPipeline
from backend.app.pipelines.hybrid_retriever import HybridRetriever
from backend.app.pipelines.retrieval_pipeline import RetrievalPipeline
from backend.app.pipelines.reranker import Reranker
from backend.app.services.document_registry import DocumentRegistry
from backend.app.services.document_service import DocumentService

from backend.evaluation.evaluation_dataset import (
    EVALUATION_DATASET,
)


class RetrievalEvaluator:
    """
    Evaluates ResearchIQ retrieval performance with and
    without cross-encoder reranking.
    """

    def __init__(
        self,
        upload_dir: str | Path = "data/uploads",
    ) -> None:

        self.upload_dir = Path(upload_dir)

        self.document_registry = DocumentRegistry()

        self.document_service = DocumentService(
            document_registry=self.document_registry,
        )

        self.embedding_pipeline = (
            self.document_service.embedding_pipeline
        )

        self.vector_store = (
            self.document_service.vector_store
        )

        self.reranker = Reranker()

        self._load_evaluation_documents()

    def _load_evaluation_documents(self) -> None:
        """
        Process all documents required by the evaluation.
        """

        filenames = {
            item["filename"]
            for item in EVALUATION_DATASET
        }

        for filename in filenames:

            file_path = (
                self.upload_dir / filename
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
        Find the registered document ID for a filename.
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
            f"Document not found in evaluation registry: "
            f"{filename}"
        )

    def _retrieve_without_reranker(
        self,
        chunks,
        question: str,
        top_k: int,
        candidate_k: int,
    ):
        """
        Retrieve using hybrid retrieval only.
        """

        hybrid_retriever = HybridRetriever(
            chunks=chunks,
            embedding_pipeline=(
                self.embedding_pipeline
            ),
            vector_store=self.vector_store,
        )

        results = hybrid_retriever.search(
            query=question,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        return [
            chunk
            for chunk, _ in results
        ]

    def _retrieve_with_reranker(
        self,
        chunks,
        question: str,
        top_k: int,
        candidate_k: int,
    ):
        """
        Retrieve using hybrid retrieval followed by
        cross-encoder reranking.
        """

        retrieval_pipeline = RetrievalPipeline(
            chunks=chunks,
            embedding_pipeline=(
                self.embedding_pipeline
            ),
            vector_store=self.vector_store,
            reranker=self.reranker,
        )

        results = retrieval_pipeline.retrieve(
            query=question,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        return [
            chunk
            for chunk, _ in results
        ]

    def _calculate_metrics(
        self,
        retrieved_chunks,
        relevant_chunks,
    ) -> tuple[float, float, float]:
        """
        Calculate Recall@K, Precision@K and
        Reciprocal Rank for one query.
        """

        retrieved_ids = [
            chunk.chunk_index
            for chunk in retrieved_chunks
        ]

        relevant_ids = set(
            relevant_chunks
        )

        relevant_retrieved = sum(
            1
            for chunk_id in retrieved_ids
            if chunk_id in relevant_ids
        )

        # -----------------------------------------------
        # Recall
        # -----------------------------------------------

        recall = (
            relevant_retrieved
            / len(relevant_ids)
            if relevant_ids
            else 0.0
        )

        recall = min(
            recall,
            1.0,
        )

        # -----------------------------------------------
        # Precision
        # -----------------------------------------------

        precision = (
            relevant_retrieved
            / len(retrieved_ids)
            if retrieved_ids
            else 0.0
        )

        # -----------------------------------------------
        # Reciprocal Rank
        # -----------------------------------------------

        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(
            retrieved_ids,
            start=1,
        ):

            if chunk_id in relevant_ids:

                reciprocal_rank = (
                    1.0 / rank
                )

                break

        return (
            recall,
            precision,
            reciprocal_rank,
        )

    def _evaluate_mode(
        self,
        use_reranker: bool,
        top_k: int,
        candidate_k: int,
    ) -> dict:
        """
        Evaluate either the baseline hybrid retriever
        or the reranked retrieval pipeline.
        """

        recall_scores = []
        precision_scores = []
        reciprocal_ranks = []

        individual_results = []

        for item in EVALUATION_DATASET:

            filename = item["filename"]
            question = item["question"]

            relevant_chunks = set(
                item["relevant_chunks"]
            )

            document_id = self._get_document_id(
                filename
            )

            chunks = (
                self.document_registry.get_document(
                    document_id
                )
            )

            if use_reranker:

                retrieved_chunks = (
                    self._retrieve_with_reranker(
                        chunks=chunks,
                        question=question,
                        top_k=top_k,
                        candidate_k=candidate_k,
                    )
                )

            else:

                retrieved_chunks = (
                    self._retrieve_without_reranker(
                        chunks=chunks,
                        question=question,
                        top_k=top_k,
                        candidate_k=candidate_k,
                    )
                )

            recall, precision, reciprocal_rank = (
                self._calculate_metrics(
                    retrieved_chunks=retrieved_chunks,
                    relevant_chunks=relevant_chunks,
                )
            )

            retrieved_ids = [
                chunk.chunk_index
                for chunk in retrieved_chunks
            ]

            recall_scores.append(
                recall
            )

            precision_scores.append(
                precision
            )

            reciprocal_ranks.append(
                reciprocal_rank
            )

            individual_results.append(
                {
                    "question": question,
                    "filename": filename,
                    "retrieved_chunks": retrieved_ids,
                    "relevant_chunks": sorted(
                        relevant_chunks
                    ),
                    "recall_at_k": recall,
                    "precision_at_k": precision,
                    "reciprocal_rank": (
                        reciprocal_rank
                    ),
                }
            )

        count = len(
            EVALUATION_DATASET
        )

        return {
            "recall_at_k": (
                sum(recall_scores) / count
            ),
            "precision_at_k": (
                sum(precision_scores) / count
            ),
            "mrr": (
                sum(reciprocal_ranks) / count
            ),
            "results": individual_results,
        }

    def evaluate(
        self,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> dict:
        """
        Compare hybrid retrieval with and without
        cross-encoder reranking.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0."
            )

        if top_k > candidate_k:
            raise ValueError(
                "top_k cannot be greater than candidate_k."
            )

        print(
            "\n" + "=" * 60
        )

        print(
            "EVALUATING WITHOUT RERANKER"
        )

        print(
            "=" * 60
        )

        without_reranker = (
            self._evaluate_mode(
                use_reranker=False,
                top_k=top_k,
                candidate_k=candidate_k,
            )
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "EVALUATING WITH RERANKER"
        )

        print(
            "=" * 60
        )

        with_reranker = (
            self._evaluate_mode(
                use_reranker=True,
                top_k=top_k,
                candidate_k=candidate_k,
            )
        )

        return {
            "top_k": top_k,
            "candidate_k": candidate_k,
            "num_examples": len(
                EVALUATION_DATASET
            ),
            "without_reranker": (
                without_reranker
            ),
            "with_reranker": (
                with_reranker
            ),
        }