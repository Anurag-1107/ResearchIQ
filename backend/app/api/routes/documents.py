from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.services.document_registry import DocumentRegistry
from backend.app.services.document_service import DocumentService


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# Shared in-memory registry for processed documents.
document_registry = DocumentRegistry()

# Document service uses the shared registry.
document_service = DocumentService(
    document_registry=document_registry,
)


UPLOAD_DIR = Path("data/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        result = document_service.process_document(
            file_path=file_path,
        )

        return {
            "message": "Document processed successfully.",
            **result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(error)}",
        )

    finally:
        await file.close()