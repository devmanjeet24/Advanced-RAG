from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def evaluate(answer, context):
    scores = model.predict([(answer, c) for c in context])

    return {
        "faithfulness": float(sum(scores)/len(scores)),
        "context_precision": float(max(scores)),
        "context_recall": float(min(scores)),
        "answer_relevance": float(sum(scores)/len(scores))
    }