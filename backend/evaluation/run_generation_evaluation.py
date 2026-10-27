from backend.evaluation.generation_evaluator import (
    GenerationEvaluator,
)


def main() -> None:
    """
    Run ResearchIQ generation and answer-grounding evaluation.
    """

    evaluator = GenerationEvaluator(
        upload_dir="data/uploads",
    )

    results = evaluator.evaluate()

    print("\n" + "=" * 60)
    print("RESEARCHIQ GENERATION EVALUATION")
    print("=" * 60)

    print(
        f"\nEvaluation examples: "
        f"{results['num_examples']}"
    )

    print(
        f"Total claims: "
        f"{results['total_claims']}"
    )

    print(
        f"Evidence refusals: "
        f"{results['evidence_refusal_count']}"
    )

    print("\n" + "-" * 60)
    print("CLAIM-LEVEL GROUNDING")
    print("-" * 60)

    print(
        f"Supported: "
        f"{results['supported_claims']} "
        f"({results['supported_claim_rate']:.2%})"
    )

    print(
        f"Partially supported: "
        f"{results['partially_supported_claims']} "
        f"({results['partially_supported_claim_rate']:.2%})"
    )

    print(
        f"Unsupported: "
        f"{results['unsupported_claims']} "
        f"({results['unsupported_claim_rate']:.2%})"
    )

    print("\n" + "-" * 60)
    print("OVERALL VERIFICATION")
    print("-" * 60)

    for status, count in (
        results[
            "overall_status_counts"
        ].items()
    ):
        print(
            f"{status}: {count}"
        )

    print("\n" + "-" * 60)
    print("INDIVIDUAL RESULTS")
    print("-" * 60)

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

        # ------------------------------------------------
        # Legitimate insufficient-evidence refusal
        # ------------------------------------------------

        if result["evidence_refusal"]:

            print(
                "Result: "
                "INSUFFICIENT EVIDENCE REFUSAL"
            )

            print(
                f"Answer: "
                f"{result['answer']}"
            )

            continue

        # ------------------------------------------------
        # Normal grounded answer
        # ------------------------------------------------

        verification = (
            result["verification"]
        )

        print(
            f"Overall status: "
            f"{verification['overall_status']}"
        )

        print(
            f"Claims: "
            f"{len(verification['claims'])}"
        )

        for claim in verification["claims"]:

            print(
                f"\n  Status: "
                f"{claim['status']}"
            )

            print(
                f"  Claim: "
                f"{claim['claim']}"
            )

            print(
                f"  Evidence: "
                f"{claim['evidence_indexes']}"
            )

            print(
                f"  Reason: "
                f"{claim['reason']}"
            )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()