SYSTEM_PROMPT = """
You are ResearchIQ, an AI research assistant.

Your task is to compare multiple research documents using ONLY
the evidence provided to you.

STRICT RULES:

1. Use only the provided evidence.
2. Do not use outside knowledge.
3. Every factual comparison must be supported by the evidence.
4. Do not invent similarities, differences, results, methods,
   conclusions, or limitations.
5. Do not infer that something is absent simply because it does
   not appear in the retrieved evidence.
6. If the evidence does not establish a difference, do not claim
   that the documents differ.
7. Clearly distinguish information belonging to each document.
8. Do not mention evidence indexes, evidence IDs, chunk IDs,
   reranker scores, or internal retrieval details.
9. Do not include citations such as [EVIDENCE 1] in the answer.
10. If the evidence is insufficient to compare a particular aspect,
    explicitly state that the available evidence is insufficient.
11. Answer the user's question directly.
12. Keep the comparison concise and well structured.
"""


def build_comparison_prompt(
    question: str,
    document_contexts: list[dict],
) -> str:
    """
    Build a prompt containing separately labeled evidence
    from multiple documents.
    """

    context_parts = []

    for document_context in document_contexts:

        context_parts.append(
            f"""
==============================
DOCUMENT: {document_context["filename"]}
==============================

{document_context["context"]}
"""
        )

    combined_context = "\n".join(
        context_parts
    )

    return f"""
Compare the provided research documents and answer the question.

QUESTION:
{question}

DOCUMENT EVIDENCE:

{combined_context}

==============================

INSTRUCTIONS:

- Use ONLY the evidence provided above.
- Compare the documents only when the evidence supports
  the comparison.
- Clearly identify which information belongs to which document.
- State similarities only when supported by evidence from
  the relevant documents.
- State differences only when the evidence explicitly
  establishes a difference.
- Do NOT treat missing evidence as evidence of absence.
- Do NOT make assumptions based on general knowledge.
- Do NOT add information that is not present in the evidence.
- Do NOT mention evidence numbers or internal retrieval details.
- Answer the question directly.
- If the evidence is insufficient for a comparison, say so.
- Use a concise structure such as:
  Similarities
  Differences
  Conclusion
  when appropriate.

Return only the user-facing comparison answer.
"""