from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional

from app.services.evaluation import evaluate
from app.services.rag_pipeline import rag_pipeline
from app.core.dependencies import get_user

router = APIRouter()


class EvalRequest(BaseModel):
    answer: str
    context: List[str]


class PipelineEvalRequest(BaseModel):
    query: str
    strategy: str = "hybrid_multi"
    filters: Optional[dict] = None
    user_filter: bool = True


@router.post("/score")
async def eval_score(req: EvalRequest):
    """Evaluate a given answer against a list of context chunks."""
    if not req.context:
        return {"error": "context list cannot be empty"}

    scores = evaluate(req.answer, req.context)
    return scores


@router.post("/pipeline")
async def eval_pipeline(req: PipelineEvalRequest, user=Depends(get_user)):
    """
    Run the full RAG pipeline on a query and evaluate the answer quality
    against the retrieved context automatically.
    """
    filters = req.filters or {}
    if req.user_filter:
        filters["user_id"] = str(user["_id"])

    result = await rag_pipeline(
        query=req.query,
        filters=filters,
        strategy=req.strategy,
    )

    if "error" in result:
        return result

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    if not sources:
        return {**result, "eval": {"error": "no sources to evaluate against"}}

    scores = evaluate(answer, sources)

    return {
        **result,
        "eval": scores,
    }