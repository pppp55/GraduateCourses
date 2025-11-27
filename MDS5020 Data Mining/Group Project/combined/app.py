"""Combined FastAPI app exposing both Subtask1 (sentiment) and Subtask2 (topic).

This file delegates to the existing implementations under `subtask1/` and
`subtask2/`. The Docker build copies the whole repository so imports like
`subtask1.sentiment_inference` and `subtask2.api_service` work without moving
models.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

# Import sentiment helpers (Subtask1)
from .subtask1.sentiment_inference import (
    load_model as load_sentiment_model,
    predict_sentiment as predict_sentiment_fn,
)

# Import topic pipeline (Subtask2)
from .subtask2 import api_service as topic_service

app = FastAPI(title="Combined News Service", version="1.0.0")

# Models loaded on startup
SENTIMENT_MODEL = None
TOPIC_PIPELINE = None


class SentimentRequest(BaseModel):
    text: Optional[str] = Field(None, description="单条新闻文本")
    news_text: Optional[str] = Field(None, description="兼容旧字段的单条新闻文本")
    news_texts: Optional[List[str]] = Field(
        None, description="多条新闻文本，至少包含一条", min_length=1
    )

    @model_validator(mode="after")
    def validate_any_text(self):
        if not self.texts():
            raise ValueError("text/news_text/news_texts 至少提供一个")
        return self

    def texts(self) -> List[str]:
        if self.news_texts:
            return self.news_texts
        for candidate in (self.text, self.news_text):
            if candidate:
                return [candidate]
        return []


class TopicRequest(BaseModel):
    text: Optional[str] = Field(None, description="新闻正文或标题")
    title: Optional[str] = Field(None, description="可选标题字段")

    @model_validator(mode="after")
    def validate_text(self):
        if not self.text and not self.title:
            raise ValueError("text 或 title 至少提供一个")
        return self

    def resolved_text(self) -> str:
        return self.text or self.title or ""


@app.on_event("startup")
async def startup_event():
    global SENTIMENT_MODEL, TOPIC_PIPELINE
    # Load sentiment model
    SENTIMENT_MODEL = load_sentiment_model()
    # topic_service already loads its pipeline at import; ensure reference
    TOPIC_PIPELINE = topic_service.pipeline


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sentiment_model_loaded": SENTIMENT_MODEL is not None,
        "topic_model_loaded": TOPIC_PIPELINE is not None,
    }


@app.post("/predict_sentiment")
async def predict_sentiment(payload: SentimentRequest):
    if SENTIMENT_MODEL is None:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded")
    try:
        results = predict_sentiment_fn(payload.texts(), model=SENTIMENT_MODEL)
        return {"count": len(results), "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_topic")
async def predict_topic(req: TopicRequest):
    try:
        text = req.resolved_text()
        pred = topic_service.pipeline.predict([text])[0]
        probs = topic_service.pipeline.predict_proba([text])[0]
        confidence = float(max(probs))
        topic_id = topic_service.TOPIC_TO_ID.get(pred, "未知")
        return {"topic": pred, "topic_id": topic_id, "probability": f"{confidence:.4f}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topics")
async def topics():
    return {"topics": topic_service.topic_labels.tolist(), "mapping": topic_service.TOPIC_TO_ID}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("combined.app:app", host="0.0.0.0", port=5724, log_level="info")
