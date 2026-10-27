# ResearchIQ

## AI-Powered Research Paper Analysis

ResearchIQ is an AI-powered research assistant that allows users to upload research papers, ask evidence-grounded questions, compare multiple papers, verify generated claims against retrieved evidence, and identify potential research gaps.

The project implements a complete Retrieval-Augmented Generation (RAG) pipeline with hybrid retrieval, cross-encoder reranking, LLM-based answer generation, and citation verification.

---

## Features

- 📄 Upload and process research papers in PDF format
- ✂️ Intelligent document chunking
- 🧠 Semantic embeddings for document retrieval
- 🔎 Hybrid retrieval for improved evidence recall
- 🤖 Cross-encoder reranking for better result ordering
- 💬 Evidence-grounded question answering
- 📚 Multi-document comparison
- 🛡️ Claim-level citation verification
- 🧩 Evidence-aware refusal when information is insufficient
- 🔬 Research gap analysis
- 📊 Retrieval evaluation using Recall@K, Precision@K, and MRR
- 📈 Generation grounding evaluation
- ⚡ FastAPI backend
- 🖥️ Streamlit frontend
- 🗄️ ChromaDB vector storage

---

# System Architecture

```text
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         │      Frontend        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │       Backend        │
                         └──────────┬───────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                                       │
                ▼                                       ▼
        Document Upload                            User Query
                │                                       │
                ▼                                       ▼
       Document Ingestion                     Hybrid Retrieval
                │                                       │
                ▼                                       ▼
           Chunking                              Candidates
                │                                       │
                ▼                                       ▼
          Embeddings                           Cross-Encoder
                │                                Reranking
                ▼                                       │
           ChromaDB                                    ▼
                │                                Top-K Evidence
                │                                       │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                              LLM Generation
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
                   Q&A        Comparison      Gap Analysis
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
                         Citation Verification
                                    │
                                    ▼
                             Final Response
```

## RAG Pipeline

The core ResearchIQ pipeline follows:

```
Research Paper PDF
       │
       ▼
Document Ingestion
       │
       ▼
Document Chunking
       │
       ▼
Embedding Generation
       │
       ▼
ChromaDB Vector Store
       │
       ▼
Hybrid Retrieval
       │
       ▼
Candidate Documents
       │
       ▼
Cross-Encoder Reranking
       │
       ▼
Top-K Evidence
       │
       ▼
Prompt Construction
       │
       ▼
LLM Generation
       │
       ▼
Generated Answer
       │
       ▼
Citation Verification
       │
       ▼
Grounded Response
```

## 1. Document Upload and Processing

ResearchIQ allows users to upload one or more research papers in PDF format.

Each uploaded document passes through a processing pipeline:

```
PDF
 │
 ▼
Text Extraction
 │
 ▼
Chunking
 │
 ▼
Embedding Generation
 │
 ▼
Vector Storage
```

Each document and chunk maintains metadata such as:

- Document ID
- Filename
- Page number
- Chunk index
- Text content

This metadata allows retrieved evidence to be traced back to the original research paper.

## 2. Document Chunking

Research papers are divided into smaller chunks before being embedded and indexed.

Each chunk retains its document-level metadata.

```
DocumentChunk
│
├── document_id
├── filename
├── page_number
├── chunk_index
└── text
```

Keeping this metadata attached to every chunk enables source-level traceability during retrieval and generation.

## 3. Embedding Generation

ResearchIQ converts document chunks into vector representations using a sentence-transformer embedding model.

The embeddings represent the semantic meaning of each chunk and are used during semantic retrieval.

The resulting vectors are stored in ChromaDB.

## 4. Hybrid Retrieval

ResearchIQ uses a hybrid retrieval approach to improve retrieval quality.

The retrieval process combines semantic and lexical retrieval signals to identify potentially relevant chunks.

```
                    User Query
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
       Semantic Retrieval   Lexical Retrieval
              │                   │
              └─────────┬─────────┘
                        │
                        ▼
                 Candidate Chunks
                        │
                        ▼
                Cross-Encoder
                   Reranking
                        │
                        ▼
                  Final Evidence
```

Hybrid retrieval is useful because semantic search can capture meaning while lexical matching can preserve important exact terminology.

## 5. Cross-Encoder Reranking

After the initial retrieval stage, ResearchIQ uses a cross-encoder to rerank candidate chunks.

The current reranker is:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker evaluates each query-document pair:

```
Question + Document Chunk
          │
          ▼
    Cross-Encoder
          │
          ▼
    Relevance Score
```

The candidates are then sorted according to their reranker scores and the highest-ranked chunks are selected as the final evidence.

This creates a two-stage retrieval architecture:

```
Stage 1:
Fast Candidate Retrieval

        ↓

Stage 2:
Cross-Encoder Reranking

        ↓

High-Quality Top-K Evidence
```

## 6. Evidence-Grounded Question Answering

Users can ask questions about an uploaded research paper.

For example:

> How does scaled dot-product attention work?

ResearchIQ retrieves relevant chunks and uses only the retrieved evidence when constructing the answer.

The workflow is:

```
User Question
     │
     ▼
Hybrid Retrieval
     │
     ▼
Candidate Chunks
     │
     ▼
Cross-Encoder Reranking
     │
     ▼
Top-K Evidence
     │
     ▼
LLM Prompt
     │
     ▼
Generated Answer
```

The generated answer is accompanied by source metadata such as:

- Document
- Page
- Chunk
- Reranker score

This makes the answer traceable to the uploaded research paper.

## 7. Citation Verification

ResearchIQ includes a separate citation verification layer.

The purpose of this component is to verify whether factual claims in the generated answer are actually supported by the retrieved evidence.

Each meaningful claim is classified as:

- SUPPORTED
- PARTIALLY_SUPPORTED
- UNSUPPORTED

For every claim, the verifier returns:

- Claim
- Verification status
- Supporting evidence indexes
- Reason

Example:

```json
{
  "claim": "Scale by 1/√d_k",
  "status": "SUPPORTED",
  "evidence_indexes": [1, 2],
  "reason": "The retrieved evidence describes the scaling operation."
}
```

The verification pipeline is:

```
Generated Answer
       │
       ▼
Extract Meaningful Claims
       │
       ▼
Compare Claims With Evidence
       │
       ▼
┌─────────────────────────────┐
│ SUPPORTED                   │
│ PARTIALLY_SUPPORTED         │
│ UNSUPPORTED                 │
└─────────────────────────────┘
       │
       ▼
Verification Result
```

This provides an additional grounding layer between the LLM and the final response.

## 8. Evidence-Aware Refusal

ResearchIQ does not force the LLM to answer when the retrieved evidence is insufficient.

Instead, it can return:

> I don't have enough information in the provided research documents
> to answer this question.

This prevents the system from presenting unsupported information as if it came from the uploaded research papers.

## 9. Multi-Document Comparison

ResearchIQ supports comparison between multiple research papers.

For example:

> How do the two papers approach language representation?

Evidence is retrieved independently from each selected document.

```
Document A
    │
    ▼
Relevant Evidence
    │
    │
Document B
    │
    ▼
Relevant Evidence
    │
    └──────────────┐
                   ▼
            Comparison Prompt
                   │
                   ▼
                   LLM
                   │
                   ▼
          Comparison Answer
```

Each document's evidence is clearly separated before being provided to the LLM.

This allows the system to identify:

- Similarities
- Differences
- Architectural differences
- Representation strategies
- Training strategies
- Computational considerations
- Methodological differences
- Limitations

The comparison prompt also instructs the model not to assume that missing information means that the two papers are different.

## 10. Research Gap Analysis

ResearchIQ can analyze multiple research papers to identify potential research gaps.

The analysis can consider:

- Common research themes
- Explicit limitations
- Future work
- Unexplored areas
- Potential research gaps
- Research opportunities
- Evidence limitations

The output is intended to identify potential research directions based on the uploaded papers rather than make definitive claims about the entire research field.

---

## Technology Stack

**Backend**
- Python
- FastAPI
- Uvicorn

**Frontend**
- Streamlit

**NLP / Machine Learning**
- Sentence Transformers
- Cross-Encoder
- Text Embeddings
- Retrieval-Augmented Generation
- Large Language Models

**Vector Database**
- ChromaDB

**LLM**
- Groq API

**Document Processing**
- PDF text extraction
- Custom document chunking

---

## Project Structure

```
ResearchIQ/
│
├── backend/
│   │
│   ├── app/
│   │   │
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── documents.py
│   │   │       ├── query.py
│   │   │       ├── comparison.py
│   │   │       └── gap_analysis.py
│   │   │
│   │   ├── pipelines/
│   │   │   ├── document_ingestion.py
│   │   │   ├── chunking.py
│   │   │   ├── embedding.py
│   │   │   ├── vector_store.py
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── retrieval_pipeline.py
│   │   │   ├── reranker.py
│   │   │   └── generation_pipeline.py
│   │   │
│   │   ├── prompts/
│   │   │   ├── rag_prompt.py
│   │   │   ├── comparison_prompt.py
│   │   │   └── citation_verification_prompt.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── document.py
│   │   │   └── query.py
│   │   │
│   │   └── services/
│   │       ├── document_service.py
│   │       ├── document_registry.py
│   │       ├── comparison_service.py
│   │       ├── citation_verification_service.py
│   │       └── llm_service.py
│   │
│   └── evaluation/
│       ├── run_evaluation.py
│       ├── run_generation_evaluation.py
│       └── ...
│
├── frontend/
│   └── app.py
│
├── data/
│   └── uploads/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ResearchIQ
```

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .researchiq-env
```

Activate it:

```bash
.researchiq-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory.

```
GROQ_API_KEY=your_groq_api_key
```

Do not commit your `.env` file to GitHub.

The repository should contain an `.env.example` file:

```
GROQ_API_KEY=
```

This shows other developers which environment variables are required without exposing your API key.

---

## Running the Application

ResearchIQ consists of a FastAPI backend and a Streamlit frontend.

### Start the Backend

From the project root:

```bash
uvicorn backend.app.main:app --reload
```

The backend will be available at:

```
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```
http://127.0.0.1:8000/docs
```

### Start the Frontend

Open another terminal.

Activate the environment:

```bash
.researchiq-env\Scripts\activate
```

Then run:

```bash
streamlit run frontend/app.py
```

The Streamlit application will open in your browser.

---

## API Endpoints

### Upload Document
```
POST /documents/upload
```
Uploads and processes a research paper.

### Query Document
```
POST /query
```
Ask a question about an uploaded research paper.

### Compare Documents
```
POST /compare
```
Compare multiple research papers using independently retrieved evidence.

### Research Gap Analysis
```
POST /gap-analysis
```
Analyze multiple papers for potential research gaps and research opportunities.

---

## Example Questions

**Question Answering**
- How does scaled dot-product attention work?
- Why is self-attention more parallelizable than recurrent models?
- What are the two pre-training objectives used by BERT?
- Why does BERT use bidirectional representations?

**Multi-Paper Comparison**
- How do the two papers approach language representation?
- How do the architectures differ?
- What are the major similarities and differences between these papers?

**Research Gap Analysis**
- What are the major limitations and unexplored research directions across these papers?
- What potential research gaps can be identified from these papers?

---

## Evaluation

ResearchIQ includes separate evaluation pipelines for retrieval and generation grounding.

### Retrieval Evaluation

The retrieval system is evaluated using:

- Recall@5
- Precision@5
- Mean Reciprocal Rank (MRR)

The current evaluation compares retrieval performance before and after cross-encoder reranking.

**Without Cross-Encoder Reranking**

| Metric | Score |
|---|---|
| Recall@5 | 79.17% |
| Precision@5 | 35.00% |
| MRR | 64.58% |

**With Cross-Encoder Reranking**

| Metric | Score |
|---|---|
| Recall@5 | 100.00% |
| Precision@5 | 45.00% |
| MRR | 81.25% |

**Improvement**

| Metric | Improvement |
|---|---|
| Recall@5 | +20.83 percentage points |
| Precision@5 | +10.00 percentage points |
| MRR | +16.67 percentage points |

These results demonstrate that cross-encoder reranking improved retrieval recall, precision, and ranking quality on the current evaluation dataset.

### Generation Grounding Evaluation

ResearchIQ also evaluates whether generated claims are supported by retrieved evidence.

Current evaluation results:

| Category | Result |
|---|---|
| Supported claims | 92.86% |
| Partially supported claims | 7.14% |
| Unsupported claims | 0.00% |

The evaluation also includes evidence-refusal cases where the system determines that the retrieved evidence is insufficient to answer a question.

### Running the Evaluations

**Retrieval Evaluation**

```bash
python -m backend.evaluation.run_evaluation
```

**Generation and Citation Verification Evaluation**

```bash
python -m backend.evaluation.run_generation_evaluation
```

### Example Output

A successful question-answering request returns information similar to:

```json
{
  "document_id": "document-id",
  "question": "How does scaled dot-product attention work?",
  "answer": "Scaled dot-product attention works by computing attention scores between queries and keys, scaling the scores, applying softmax, and using the resulting weights to combine the values.",
  "sources": [
    {
      "document_id": "document-id",
      "filename": "Research_paper.pdf",
      "page_number": 4,
      "chunk_index": 13,
      "reranker_score": 8.326
    }
  ]
}
```

The exact response depends on the uploaded document and retrieved evidence.

---

## Design Principles

ResearchIQ is built around three main principles.

### 1. Grounded Generation

The LLM should generate answers based on retrieved evidence rather than unsupported external information.

### 2. Traceability

Every retrieved chunk maintains document and page metadata.

This allows answers to be traced back to the source research paper.

### 3. Evidence-Aware Assistance

The system distinguishes between information that is:

```
Retrieved Evidence
       │
       ▼
Supported Claims
       │
       ▼
Generated Answer
```

and information that cannot be supported by the available evidence.

When evidence is insufficient, the system can refuse to provide an answer rather than invent information.

---

## Limitations

ResearchIQ is currently an MVP / portfolio project.

### In-Memory Document Registry

The current document registry stores processed document chunks in memory.

Therefore:

```
Server Restart
      │
      ▼
In-Memory Documents Lost
```

Persistent document storage would be required for a production system.

### Retrieval Limitations

Retrieval quality depends on:

- Chunking strategy
- Embedding model
- Lexical retrieval
- Candidate selection
- Cross-encoder performance

### LLM Limitations

Although citation verification provides an additional grounding layer, LLM-generated answers can still contain partially supported statements.

### Research Gap Limitations

Research gap analysis is based only on the uploaded papers and retrieved evidence.

Therefore, identified gaps should be treated as potential research opportunities rather than definitive statements about the complete research literature.

### Evaluation Dataset

The current evaluation dataset is relatively small.

A larger and more diverse evaluation dataset would provide stronger evidence of general system performance.

### PDF Processing

The current implementation primarily focuses on textual content extracted from PDFs.

Complex figures, tables, mathematical equations, scanned documents, and unusual PDF layouts may require additional processing.

---

## Future Improvements

Potential future improvements include:

- Persistent document storage
- User authentication
- Document collections
- Document deletion and management
- Document deduplication
- Improved chunking strategies
- Adaptive chunking
- Query expansion
- Query decomposition
- Improved hybrid retrieval
- Advanced reranking models
- Better citation formatting
- Claim-level source highlighting
- Larger evaluation datasets
- Automated RAG evaluation
- Figure and table understanding
- Mathematical equation extraction
- Multimodal PDF analysis
- Support for additional document formats
- Literature-scale research gap analysis

---

## Project Highlights

ResearchIQ is designed as a complete RAG-based research assistant rather than a simple LLM wrapper.

The system combines:

```
PDF Processing
      │
      ▼
Document Chunking
      │
      ▼
Embeddings
      │
      ▼
Vector Database
      │
      ▼
Hybrid Retrieval
      │
      ▼
Cross-Encoder Reranking
      │
      ▼
LLM Generation
      │
      ▼
Citation Verification
      │
      ▼
Research Analysis
```

The project also includes quantitative evaluation to measure the impact of cross-encoder reranking and claim-level grounding evaluation to measure how well generated responses are supported by retrieved evidence.

---

## Results at a Glance

### Retrieval

**Recall@5**
79.17% → 100.00%

**Precision@5**
35.00% → 45.00%

**MRR**
64.58% → 81.25%

The comparison represents retrieval without versus with cross-encoder reranking on the current evaluation dataset.

### Generation Grounding

```
Supported claims:          92.86%
Partially supported:        7.14%
Unsupported claims:         0.00%
```

These results are based on the current evaluation dataset and should not be interpreted as universal benchmark performance.

---

## Development Status

**Completed**
- [x] PDF document ingestion
- [x] Document chunking
- [x] Embedding generation
- [x] ChromaDB vector storage
- [x] Hybrid retrieval
- [x] Cross-encoder reranking
- [x] RAG-based question answering
- [x] Citation verification
- [x] Evidence-aware refusal
- [x] Multi-document comparison
- [x] Research gap analysis
- [x] Retrieval evaluation
- [x] Generation grounding evaluation
- [x] FastAPI backend
- [x] Streamlit frontend
- [x] End-to-end integration testing

---

## Security

API keys and other secrets should never be committed to the repository.

Make sure `.env` is included in `.gitignore`.

Example:

```
.env
__pycache__/
*.pyc
.researchiq-env/
```

Before pushing the project to GitHub, verify that no API keys or other secrets are present in tracked files.

---

## Author

**Anurag Gupta**

ResearchIQ is a portfolio project demonstrating the design and implementation of an evidence-grounded research paper analysis system using modern RAG techniques.

---

## License

This project is intended for educational and portfolio purposes.