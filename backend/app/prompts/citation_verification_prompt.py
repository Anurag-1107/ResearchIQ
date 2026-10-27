SYSTEM_PROMPT = """
You are ResearchIQ, an AI research assistant that verifies
whether an answer is supported by retrieved evidence.

Use ONLY the provided evidence.

For every meaningful factual claim:
- SUPPORTED if directly supported.
- PARTIALLY_SUPPORTED if only partly supported.
- UNSUPPORTED if not supported.

Return ONLY valid JSON.

The JSON MUST contain:
{
    "claims": [
        {
            "claim": "short claim",
            "status": "SUPPORTED",
            "evidence_indexes": [1],
            "reason": "short reason"
        }
    ],
    "overall_status": "SUPPORTED"
}

Allowed statuses:
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED

Keep claims very short.
Keep reasons under 10 words.
Do not repeat unnecessary wording.
Do not include markdown.
Do not include code fences.
Return JSON only.
"""


def build_verification_prompt(
    answer: str,
    evidence: list[dict],
) -> str:
    """
    Build a compact verification prompt.
    """

    evidence_parts = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        evidence_parts.append(
            f"""
[EVIDENCE {index}]
Page: {item["page_number"]}
Chunk: {item["chunk_index"]}
Text:
{item["text"]}
"""
        )

    combined_evidence = "\n".join(
        evidence_parts
    )

    return f"""
Verify this answer using ONLY the evidence.

ANSWER:
{answer}

EVIDENCE:
{combined_evidence}

Instructions:
1. Extract meaningful factual claims from the answer.
2. Classify each claim.
3. Give evidence indexes.
4. Give a reason of 10 words or fewer.
5. Give the overall status.

Return ONLY the required JSON object.
"""