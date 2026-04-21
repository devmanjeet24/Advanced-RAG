def document_model(doc):
    return {
        "id": str(doc["_id"]),
        "filename": doc["filename"],
        "metadata": doc.get("metadata", {})
    }