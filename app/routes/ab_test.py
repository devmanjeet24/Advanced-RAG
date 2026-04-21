from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
import time

from app.services.rag_pipeline import rag_pipeline
from app.core.dependencies import get_user

router = APIRouter()


class ABRequest(BaseModel):
    query: str
    strategy_a: str = "basic"
    strategy_b: str = "hybrid_multi"
    top_k: int = 5
    filters: Optional[dict] = None
    user_filter: bool = True


@router.post("/")
async def ab_test(req: ABRequest, user=Depends(get_user)):
    """
    Run the same query against two different RAG strategies and compare results.

    Strategy options:
      basic | hyde | multi_query | decompose | step_back | hybrid_multi
    """
    filters = req.filters or {}
    if req.user_filter:
        filters["user_id"] = str(user["_id"])

    # ── Run Strategy A ──
    t0 = time.perf_counter()
    result_a = await rag_pipeline(
        query=req.query,
        filters=filters,
        strategy=req.strategy_a,
        top_k=req.top_k,
    )
    time_a = round(time.perf_counter() - t0, 3)

    # ── Run Strategy B ──
    t0 = time.perf_counter()
    result_b = await rag_pipeline(
        query=req.query,
        filters=filters,
        strategy=req.strategy_b,
        top_k=req.top_k,
    )
    time_b = round(time.perf_counter() - t0, 3)

    return {
        "query": req.query,
        "A": {
            "strategy": req.strategy_a,
            "answer": result_a.get("answer"),
            "sources_count": len(result_a.get("sources", [])),
            "latency_s": time_a,
            "error": result_a.get("error"),
        },
        "B": {
            "strategy": req.strategy_b,
            "answer": result_b.get("answer"),
            "sources_count": len(result_b.get("sources", [])),
            "latency_s": time_b,
            "error": result_b.get("error"),
        },
        "comparison": {
            "faster": req.strategy_a if time_a < time_b else req.strategy_b,
            "latency_diff_s": round(abs(time_a - time_b), 3),
        }
    }