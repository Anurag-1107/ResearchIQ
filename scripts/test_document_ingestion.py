from backend.app.pipelines.document_ingestion import DocumentIngestionPipeline


def main():
    pipeline = DocumentIngestionPipeline()

    extracted_document = pipeline.extract_pdf(
        "data/raw/Research_paper.pdf"
    )

    print("\n=== DOCUMENT METADATA ===")
    print(f"Document ID: {extracted_document.metadata.document_id}")
    print(f"Filename: {extracted_document.metadata.filename}")
    print(f"Page Count: {extracted_document.metadata.page_count}")

    print("\n=== PAGE PREVIEW ===")

    for page in extracted_document.pages[:3]:
        preview = page.text[:500]

        print(f"\n--- Page {page.page_number} ---")
        print(preview)

    print("\n=== EXTRACTION SUCCESSFUL ===")


if __name__ == "__main__":
    main()