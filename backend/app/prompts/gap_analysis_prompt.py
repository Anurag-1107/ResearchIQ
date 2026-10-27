SYSTEM_PROMPT = """
You are ResearchIQ, an AI research assistant specializing in
research literature analysis.

Your task is to identify POTENTIAL research gaps from the
provided research papers.

IMPORTANT:
- Use ONLY the provided evidence.
- Do not claim that a gap is scientifically proven.
- Describe findings as potential research gaps or opportunities.
- Do not invent missing information.
- Do not assume that something is a gap simply because it is
  not mentioned in the provided evidence.
- Distinguish clearly between:
  1. limitations explicitly stated by the papers,
  2. future work explicitly suggested by the papers,
  3. potential gaps inferred from patterns across the evidence.
- Every potential gap must be supported by evidence from the
  provided documents.
- If the evidence is insufficient, explicitly say so.
- Do not mention internal retrieval details such as reranker
  scores or chunk indices.
"""


def build_gap_analysis_prompt(
    document_contexts: list[dict],
) -> str:
    """
    Build a prompt for identifying potential research gaps
    across multiple research papers.
    """

    context_parts = []

    for document_context in document_contexts:

        context_parts.append(
            f"""
==============================
DOCUMENT: {document_context["filename"]}
DOCUMENT ID: {document_context["document_id"]}
==============================

{document_context["context"]}
"""
        )

    combined_context = "\n".join(context_parts)

    return f"""
Analyze the following research papers and identify potential
research gaps based only on the supplied evidence.

RESEARCH PAPER EVIDENCE:

{combined_context}

Provide the analysis using the following structure:

## 1. Common Research Themes

Briefly describe the major research themes shared across
the papers.

## 2. Explicit Limitations

List limitations that the papers themselves explicitly
identify.

For each limitation, mention which paper discusses it.

## 3. Explicit Future Work

List future research directions explicitly suggested
by the papers.

For each direction, mention which paper suggests it.

## 4. Potential Research Gaps

Identify potential research gaps that emerge from patterns
across the provided papers.

For each potential gap:

- State the potential gap clearly.
- Explain the evidence or pattern behind it.
- Identify the papers supporting that observation.
- Clearly label it as a POTENTIAL research gap.

Do not present these as scientifically proven gaps.

## 5. Research Opportunities

Based strictly on the evidence, describe possible areas
that researchers could investigate further.

Avoid inventing specific experiments or claims that are
not supported by the papers.

## 6. Evidence Limitations

Clearly explain what the provided evidence does NOT allow
you to conclude.

Remember:

A potential research gap is an observation derived from
the available literature evidence, not a scientifically
validated discovery.
"""