from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import shutil, os, uuid
from datetime import datetime

from app.utils.file_loader import load
from app.services.parent_child import parent_child
from app.services.vectorstore import get_vectorstore
from app.core.dependencies import get_user
from app.db.mongodb import db

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = [".pdf", ".txt", ".docx", ".md"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_file(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type")

    return ext


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user=Depends(get_user)
):
    try:
        # ✅ Validate file
        ext = validate_file(file)

        # ✅ Unique filename (avoid overwrite)
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{file.filename}"
        path = os.path.join(UPLOAD_DIR, filename)

        # ✅ Save file
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ✅ Load text
        text = load(path)

        if not text or len(text.strip()) == 0:
            raise HTTPException(400, "Empty document")

        # ✅ Base metadata
        base_metadata = {
            "user_id": str(user["_id"]),
            "filename": file.filename,
            "file_id": unique_id,
            "uploaded_at": datetime.utcnow().isoformat()
        }

        # ✅ Parent-child chunking
        chunks = parent_child(text, base_metadata)

        if len(chunks) == 0:
            raise HTTPException(400, "Chunking failed")

        # ✅ Add chunk_id + enrich metadata
        enriched_chunks = []
        for i, c in enumerate(chunks):
            enriched_chunks.append({
                "text": c["text"],
                "metadata": {
                    **c["metadata"],
                    "chunk_id": i,
                    "length": len(c["text"])
                }
            })

        # ✅ Store in Vector DB
        vs = get_vectorstore()

        vs.add_texts(
            texts=[c["text"] for c in enriched_chunks],
            metadatas=[c["metadata"] for c in enriched_chunks]
        )

        # vs.persist() 

        # ✅ Store document metadata in Mongo
        await db.documents.insert_one({
            "file_id": unique_id,
            "filename": file.filename,
            "user_id": str(user["_id"]),
            "total_chunks": len(enriched_chunks),
            "created_at": datetime.utcnow()
        })

        # ✅ Store chunks in Mongo (IMPORTANT for filtering + eval)
        await db.chunks.insert_many(enriched_chunks)

        return {
            "status": "success",
            "file_id": unique_id,
            "filename": file.filename,
            "chunks_created": len(enriched_chunks)
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )