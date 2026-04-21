from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from app.services.rag_pipeline import rag_pipeline
from app.services.llm import generate_answer_stream
from app.services.vectorstore import get_vectorstore
from app.services.query_transform import multi_query, hyde, decompose, step_back
from app.core.dependencies import get_user

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    strategy: str = "hybrid_multi"   # basic | hyde | multi_query | decompose | step_back | hybrid_multi
    top_k: int = 5
    filters: Optional[dict] = None   # e.g. {"user_id": "...", "file_id": "..."}
    user_filter: bool = True          # if True, auto-adds user_id to filters


class StreamRequest(BaseModel):
    query: str
    strategy: str = "hybrid_multi"
    filters: Optional[dict] = None
    user_filter: bool = True


# ── Main query endpoint ──────────────────────────────────────────────────────
@router.post("/query")
async def query(req: QueryRequest, user=Depends(get_user)):
    filters = req.filters or {}

    if req.user_filter:
        filters["user_id"] = str(user["_id"])

    result = await rag_pipeline(
        query=req.query,
        filters=filters,
        strategy=req.strategy,
        top_k=req.top_k,
    )
    return result


# ── Streaming endpoint ───────────────────────────────────────────────────────
@router.post("/stream")
async def stream(req: StreamRequest, user=Depends(get_user)):
    filters = req.filters or {}

    if req.user_filter:
        filters["user_id"] = str(user["_id"])

    # First, get the RAG result
    result = await rag_pipeline(
        query=req.query,
        filters=filters,
        strategy=req.strategy,
    )

    if "error" in result:
        return result

    # Stream the answer token by token
    context = "\n\n".join(result.get("sources", []))
    final_prompt = (
        f"Answer the following question using ONLY the context provided below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.query}"
    )

    def gen():
        for chunk in generate_answer_stream(final_prompt):
            yield chunk

    return StreamingResponse(gen(), media_type="text/plain")


# ── Query transform preview ──────────────────────────────────────────────────
@router.post("/transform")
async def transform_query(
    q: str = Query(..., description="The query to transform"),
    strategy: str = Query("multi_query", description="hyde | multi_query | decompose | step_back"),
):
    """Preview how different query strategies transform a query (no retrieval)."""
    if strategy == "hyde":
        return {"strategy": "hyde", "result": hyde(q)}
    elif strategy == "multi_query":
        return {"strategy": "multi_query", "result": multi_query(q)}
    elif strategy == "decompose":
        return {"strategy": "decompose", "result": decompose(q)}
    elif strategy == "step_back":
        return {"strategy": "step_back", "result": step_back(q)}
    else:
        return {"error": f"Unknown strategy: {strategy}"}