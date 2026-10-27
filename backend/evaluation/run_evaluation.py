from backend.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)


def print_mode_results(
    title: str,
    results: dict,
    top_k: int,
) -> None:
    """
    Print evaluation results for one retrieval mode.
    """

    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)

    print(
        f"Recall@{top_k}: "
        f"{results['recall_at_k']:.4f}"
    )

    print(
        f"Precision@{top_k}: "
        f"{results['precision_at_k']:.4f}"
    )

    print(
        f"MRR: "
        f"{results['mrr']:.4f}"
    )

    print("\nIndividual Results")

    for index, result in enumerate(
        results["results"],
        start=1,
    ):

        print(
            f"\n[{index}] "
            f"{result['question']}"
        )

        print(
            f"Document: "
            f"{result['filename']}"
        )

        print(
            f"Relevant chunks: "
            f"{result['relevant_chunks']}"
        )

        print(
            f"Retrieved chunks: "
            f"{result['retrieved_chunks']}"
        )

        print(
            f"Recall@{top_k}: "
            f"{result['recall_at_k']:.4f}"
        )

        print(
            f"Precision@{top_k}: "
            f"{result['precision_at_k']:.4f}"
        )

        print(
            f"Reciprocal Rank: "
            f"{result['reciprocal_rank']:.4f}"
        )


def main() -> None:
    """
    Run the ResearchIQ retrieval evaluation.

    Compares:
        1. Hybrid retrieval without reranking
        2. Hybrid retrieval with cross-encoder reranking
    """

    evaluator = RetrievalEvaluator(
        upload_dir="data/uploads",
    )

    results = evaluator.evaluate(
        top_k=5,
        candidate_k=10,
    )

    top_k = results["top_k"]

    without_reranker = (
        results["without_reranker"]
    )

    with_reranker = (
        results["with_reranker"]
    )

    # ========================================================
    # Header
    # ========================================================

    print("\n" + "=" * 60)
    print("RESEARCHIQ RETRIEVAL EVALUATION")
    print("=" * 60)

    print(
        f"\nEvaluation examples: "
        f"{results['num_examples']}"
    )

    print(
        f"Top-K: "
        f"{top_k}"
    )

    print(
        f"Candidate-K: "
        f"{results['candidate_k']}"
    )

    # ========================================================
    # Without reranker
    # ========================================================

    print_mode_results(
        title="WITHOUT RERANKER",
        results=without_reranker,
        top_k=top_k,
    )

    # ========================================================
    # With reranker
    # ========================================================

    print_mode_results(
        title="WITH CROSS-ENCODER RERANKER",
        results=with_reranker,
        top_k=top_k,
    )

    # ========================================================
    # Comparison
    # ========================================================

    recall_improvement = (
        with_reranker["recall_at_k"]
        - without_reranker["recall_at_k"]
    )

    precision_improvement = (
        with_reranker["precision_at_k"]
        - without_reranker["precision_at_k"]
    )

    mrr_improvement = (
        with_reranker["mrr"]
        - without_reranker["mrr"]
    )

    print("\n" + "=" * 60)
    print("RERANKER COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Metric':<20}"
        f"{'Without':>15}"
        f"{'With':>15}"
        f"{'Change':>15}"
    )

    print("-" * 65)

    print(
        f"{'Recall@' + str(top_k):<20}"
        f"{without_reranker['recall_at_k']:>15.4f}"
        f"{with_reranker['recall_at_k']:>15.4f}"
        f"{recall_improvement:>+15.4f}"
    )

    print(
        f"{'Precision@' + str(top_k):<20}"
        f"{without_reranker['precision_at_k']:>15.4f}"
        f"{with_reranker['precision_at_k']:>15.4f}"
        f"{precision_improvement:>+15.4f}"
    )

    print(
        f"{'MRR':<20}"
        f"{without_reranker['mrr']:>15.4f}"
        f"{with_reranker['mrr']:>15.4f}"
        f"{mrr_improvement:>+15.4f}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()