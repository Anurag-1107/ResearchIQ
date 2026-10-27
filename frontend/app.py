import httpx
import streamlit as st


BACKEND_URL = "http://127.0.0.1:8000"


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="ResearchIQ",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# Session State
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "page_count" not in st.session_state:
    st.session_state.page_count = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = None


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("🧠 ResearchIQ")

    st.caption(
        "Research Intelligence Platform"
    )

    st.divider()

    # --------------------------------------------------------
    # System Status
    # --------------------------------------------------------

    st.subheader("System Status")

    try:

        response = httpx.get(
            f"{BACKEND_URL}/health",
            timeout=5.0,
        )

        response.raise_for_status()

        health_data = response.json()

        st.success("Backend Connected")

        st.caption(
            f"{health_data['service']} — "
            f"{health_data['status']}"
        )

    except httpx.HTTPError:

        st.error("Backend Disconnected")

    st.divider()

    # --------------------------------------------------------
    # Uploaded Documents
    # --------------------------------------------------------

    st.subheader("Uploaded Documents")

    if st.session_state.documents:

        for document_id, document in (
            st.session_state.documents.items()
        ):

            st.success(
                document["filename"]
            )

            st.caption(
                f"Pages: {document['page_count']} | "
                f"Chunks: {document['chunk_count']}"
            )

    else:

        st.info(
            "No documents uploaded yet."
        )


# ============================================================
# Main Header
# ============================================================

st.title("🧠 ResearchIQ")

st.subheader(
    "AI-powered research paper analysis"
)

st.write(
    "Upload research papers, ask grounded questions, "
    "compare papers, and identify potential research gaps "
    "using evidence from the uploaded documents."
)

st.divider()


# ============================================================
# Document Upload
# ============================================================

st.header("📄 Upload Research Papers")

st.write(
    "Upload one or more research papers to analyze them "
    "using ResearchIQ."
)

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)


if uploaded_files:

    if st.button(
        "Process Documents",
        type="primary",
    ):

        for uploaded_file in uploaded_files:

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }

                    response = httpx.post(
                        f"{BACKEND_URL}/documents/upload",
                        files=files,
                        timeout=120.0,
                    )

                    response.raise_for_status()

                    result = response.json()

                    document_id = result[
                        "document_id"
                    ]

                    st.session_state.documents[
                        document_id
                    ] = {
                        "filename": result[
                            "filename"
                        ],
                        "page_count": result[
                            "page_count"
                        ],
                        "chunk_count": result[
                            "chunk_count"
                        ],
                    }

                    # Most recently uploaded document
                    # becomes the active Q&A document.
                    st.session_state.document_id = (
                        document_id
                    )

                    st.session_state.filename = (
                        result["filename"]
                    )

                    st.session_state.page_count = (
                        result["page_count"]
                    )

                    st.session_state.chunk_count = (
                        result["chunk_count"]
                    )

                    st.success(
                        f"{uploaded_file.name} "
                        "processed successfully."
                    )

                except httpx.HTTPStatusError as error:

                    st.error(
                        f"Failed to process "
                        f"{uploaded_file.name}: "
                        f"{error.response.text}"
                    )

                except httpx.HTTPError as error:

                    st.error(
                        f"Could not connect to backend "
                        f"while processing "
                        f"{uploaded_file.name}: "
                        f"{error}"
                    )

        st.rerun()


# ============================================================
# Active Document Selection
# ============================================================

if st.session_state.documents:

    st.divider()

    st.subheader("📌 Select Document for Q&A")

    document_options = {
        document["filename"]: document_id
        for document_id, document
        in st.session_state.documents.items()
    }

    selected_filename = st.selectbox(
        "Choose a research paper",
        options=list(
            document_options.keys()
        ),
        index=0,
        key="active_document_selector",
    )

    selected_document_id = document_options[
        selected_filename
    ]

    st.session_state.document_id = (
        selected_document_id
    )

    selected_document = st.session_state.documents[
        selected_document_id
    ]

    st.session_state.filename = (
        selected_document["filename"]
    )

    st.session_state.page_count = (
        selected_document["page_count"]
    )

    st.session_state.chunk_count = (
        selected_document["chunk_count"]
    )

    st.info(
        f"Active document: "
        f"**{selected_document['filename']}** "
        f"— {selected_document['page_count']} pages, "
        f"{selected_document['chunk_count']} chunks"
    )


# ============================================================
# ASK RESEARCHIQ
# ============================================================

if st.session_state.document_id:

    st.divider()

    st.header("🔍 Ask ResearchIQ")

    st.write(
        "Ask a question about the currently active "
        "research paper."
    )

    question = st.text_area(
        "What would you like to know?",
        placeholder=(
            "Example: How does scaled dot-product "
            "attention work?"
        ),
        height=120,
    )

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        top_k = st.slider(
            "Evidence chunks",
            min_value=1,
            max_value=10,
            value=5,
        )

    with col2:

        st.write("")

        ask_question = st.button(
            "Ask ResearchIQ",
            type="primary",
            use_container_width=True,
        )

    if ask_question:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document and "
                "generating an answer..."
            ):

                try:

                    payload = {
                        "document_id": (
                            st.session_state.document_id
                        ),
                        "question": question,
                        "top_k": top_k,
                    }

                    response = httpx.post(
                        f"{BACKEND_URL}/query",
                        json=payload,
                        timeout=120.0,
                    )

                    response.raise_for_status()

                    result = response.json()

                    # ====================================================
                    # Answer
                    # ====================================================

                    st.divider()

                    st.header(
                        "🧠 ResearchIQ Answer"
                    )

                    st.markdown(
                        result["answer"]
                    )

                    # ====================================================
                    # Citation Verification
                    # ====================================================

                    verification = result.get(
                        "verification"
                    )

                    if verification:

                        st.divider()

                        st.header(
                            "🛡️ Citation Verification"
                        )

                        overall_status = (
                            verification.get(
                                "overall_status",
                                "VERIFICATION_FAILED",
                            )
                        )

                        if overall_status == "SUPPORTED":

                            st.success(
                                "✅ Answer is supported "
                                "by the retrieved evidence."
                            )

                        elif (
                            overall_status
                            == "PARTIALLY_SUPPORTED"
                        ):

                            st.warning(
                                "⚠️ Answer is only partially "
                                "supported by the retrieved evidence."
                            )

                        elif overall_status == "UNSUPPORTED":

                            st.error(
                                "❌ The answer contains claims "
                                "that are not adequately supported."
                            )

                        else:

                            st.info(
                                "ℹ️ Citation verification "
                                "could not be completed."
                            )

                        claims = verification.get(
                            "claims",
                            [],
                        )

                        if claims:

                            st.subheader(
                                "Claim Verification"
                            )

                            for index, claim in enumerate(
                                claims,
                                start=1,
                            ):

                                status = claim.get(
                                    "status",
                                    "UNKNOWN",
                                )

                                claim_text = claim.get(
                                    "claim",
                                    "",
                                )

                                reason = claim.get(
                                    "reason",
                                    "",
                                )

                                evidence_indexes = (
                                    claim.get(
                                        "evidence_indexes",
                                        [],
                                    )
                                )

                                # ----------------------------------------
                                # Claim Status
                                # ----------------------------------------

                                if status == "SUPPORTED":

                                    st.success(
                                        f"✅ Claim {index}: "
                                        f"{claim_text}"
                                    )

                                elif (
                                    status
                                    == "PARTIALLY_SUPPORTED"
                                ):

                                    st.warning(
                                        f"⚠️ Claim {index}: "
                                        f"{claim_text}"
                                    )

                                elif status == "UNSUPPORTED":

                                    st.error(
                                        f"❌ Claim {index}: "
                                        f"{claim_text}"
                                    )

                                else:

                                    st.info(
                                        f"Claim {index}: "
                                        f"{claim_text}"
                                    )

                                # ----------------------------------------
                                # Evidence References
                                # ----------------------------------------

                                if evidence_indexes:

                                    evidence_text = (
                                        ", ".join(
                                            f"Evidence {item}"
                                            for item
                                            in evidence_indexes
                                        )
                                    )

                                    st.caption(
                                        f"Supporting evidence: "
                                        f"{evidence_text}"
                                    )

                                # ----------------------------------------
                                # Verification Reason
                                # ----------------------------------------

                                if reason:

                                    st.caption(
                                        f"Reason: {reason}"
                                    )

                    # ====================================================
                    # Supporting Sources
                    # ====================================================

                    st.divider()

                    st.header(
                        "📚 Supporting Sources"
                    )

                    for index, source in enumerate(
                        result["sources"],
                        start=1,
                    ):

                        with st.expander(
                            f"Source {index} — "
                            f"{source['filename']} — "
                            f"Page "
                            f"{source['page_number']}",
                            expanded=(
                                index == 1
                            ),
                        ):

                            st.write(
                                f"**File:** "
                                f"{source['filename']}"
                            )

                            st.write(
                                f"**Page:** "
                                f"{source['page_number']}"
                            )

                            st.write(
                                f"**Chunk:** "
                                f"{source['chunk_index']}"
                            )

                            st.caption(
                                f"Reranker Score: "
                                f"{source['reranker_score']:.4f}"
                            )

                except httpx.HTTPStatusError as error:

                    st.error(
                        f"Request failed: "
                        f"{error.response.text}"
                    )

                except httpx.HTTPError as error:

                    st.error(
                        f"Could not connect to backend: "
                        f"{error}"
                    )


# ============================================================
# COMPARE RESEARCH PAPERS
# ============================================================

if len(st.session_state.documents) >= 2:

    st.divider()

    st.header("📊 Compare Research Papers")

    st.write(
        "Select two or more research papers and ask "
        "ResearchIQ to compare them using evidence "
        "retrieved independently from each document."
    )

    document_options = {
        document["filename"]: document_id
        for document_id, document
        in st.session_state.documents.items()
    }

    selected_compare_filenames = st.multiselect(
        "Select papers to compare",
        options=list(
            document_options.keys()
        ),
        default=list(
            document_options.keys()
        )[:2],
        key="compare_documents",
    )

    compare_top_k = st.slider(
        "Evidence chunks per paper",
        min_value=1,
        max_value=10,
        value=5,
        key="compare_top_k",
    )

    comparison_question = st.text_area(
        "What would you like to compare?",
        placeholder=(
            "Example: Compare the main approaches "
            "used in these two papers."
        ),
        height=100,
        key="comparison_question",
    )

    compare_button = st.button(
        "📊 Compare Papers",
        type="primary",
        use_container_width=True,
    )

    if compare_button:

        if len(selected_compare_filenames) < 2:

            st.warning(
                "Please select at least two papers."
            )

        elif not comparison_question.strip():

            st.warning(
                "Please enter what you would like "
                "to compare."
            )

        else:

            selected_compare_ids = [
                document_options[filename]
                for filename
                in selected_compare_filenames
            ]

            payload = {
                "document_ids": selected_compare_ids,
                "question": comparison_question,
                "top_k": compare_top_k,
            }

            with st.spinner(
                "Comparing the selected research papers..."
            ):

                try:

                    response = httpx.post(
                        f"{BACKEND_URL}/compare",
                        json=payload,
                        timeout=180.0,
                    )

                    response.raise_for_status()

                    result = response.json()

                    st.divider()

                    st.header(
                        "📊 ResearchIQ Comparison"
                    )

                    st.markdown(
                        result["answer"]
                    )

                    st.divider()

                    st.header(
                        "📚 Comparison Evidence"
                    )

                    for index, source in enumerate(
                        result["sources"],
                        start=1,
                    ):

                        with st.expander(
                            f"Source {index} — "
                            f"{source['filename']} — "
                            f"Page "
                            f"{source['page_number']}",
                            expanded=(
                                index <= 2
                            ),
                        ):

                            st.write(
                                f"**File:** "
                                f"{source['filename']}"
                            )

                            st.write(
                                f"**Page:** "
                                f"{source['page_number']}"
                            )

                            st.write(
                                f"**Chunk:** "
                                f"{source['chunk_index']}"
                            )

                            st.caption(
                                f"Reranker Score: "
                                f"{source['reranker_score']:.4f}"
                            )

                except httpx.HTTPStatusError as error:

                    st.error(
                        f"Comparison failed: "
                        f"{error.response.text}"
                    )

                except httpx.HTTPError as error:

                    st.error(
                        f"Could not connect to backend "
                        f"while comparing papers: "
                        f"{error}"
                    )


# ============================================================
# RESEARCH GAP ANALYSIS
# ============================================================

if len(st.session_state.documents) >= 2:

    st.divider()

    st.header("🔬 Research Gap Analysis")

    st.write(
        "Analyze multiple research papers to identify "
        "limitations, future directions, and "
        "evidence-backed potential research gaps."
    )

    gap_document_options = {
        document["filename"]: document_id
        for document_id, document
        in st.session_state.documents.items()
    }

    selected_gap_filenames = st.multiselect(
        "Select papers for gap analysis",
        options=list(
            gap_document_options.keys()
        ),
        default=list(
            gap_document_options.keys()
        )[:2],
        key="gap_documents",
    )

    gap_top_k = st.slider(
        "Evidence chunks per paper",
        min_value=1,
        max_value=10,
        value=5,
        key="gap_top_k",
    )

    analyze_gaps = st.button(
        "🔬 Analyze Research Gaps",
        type="primary",
        use_container_width=True,
    )

    if analyze_gaps:

        if len(selected_gap_filenames) < 2:

            st.warning(
                "Please select at least two "
                "research papers."
            )

        else:

            selected_gap_ids = [
                gap_document_options[filename]
                for filename
                in selected_gap_filenames
            ]

            payload = {
                "document_ids": selected_gap_ids,
                "top_k": gap_top_k,
            }

            with st.spinner(
                "Analyzing research papers and "
                "identifying potential gaps..."
            ):

                try:

                    response = httpx.post(
                        f"{BACKEND_URL}/gap-analysis",
                        json=payload,
                        timeout=180.0,
                    )

                    response.raise_for_status()

                    result = response.json()

                    st.divider()

                    st.header(
                        "🧪 Potential Research Gaps"
                    )

                    st.markdown(
                        result["answer"]
                    )

                    st.divider()

                    st.header(
                        "📚 Gap Analysis Evidence"
                    )

                    for index, source in enumerate(
                        result["sources"],
                        start=1,
                    ):

                        with st.expander(
                            f"Evidence {index} — "
                            f"{source['filename']} — "
                            f"Page "
                            f"{source['page_number']}",
                            expanded=(
                                index <= 2
                            ),
                        ):

                            st.write(
                                f"**File:** "
                                f"{source['filename']}"
                            )

                            st.write(
                                f"**Page:** "
                                f"{source['page_number']}"
                            )

                            st.write(
                                f"**Chunk:** "
                                f"{source['chunk_index']}"
                            )

                            st.caption(
                                f"Reranker Score: "
                                f"{source['reranker_score']:.4f}"
                            )

                except httpx.HTTPStatusError as error:

                    st.error(
                        f"Gap analysis failed: "
                        f"{error.response.text}"
                    )

                except httpx.HTTPError as error:

                    st.error(
                        f"Could not connect to backend "
                        f"while analyzing gaps: "
                        f"{error}"
                    )