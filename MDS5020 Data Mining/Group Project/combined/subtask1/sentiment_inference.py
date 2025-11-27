"""Reusable helper for Subtask 1 sentiment predictions.

This module loads the best-performing pipeline produced during training
(`sentiment_best_pipeline.pkl`) and exposes a lightweight prediction API that
returns both labels and probabilities. It is designed to be imported by the
upcoming FastAPI/Flask service as well as exercised directly from the command
line for quick smoke tests prior to containerizing the solution.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import emoji
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:  # Ensure the required corpus exists before processing requests
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:  # pragma: no cover - executed only in new environments
    import nltk

    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

STEMMER = PorterStemmer()

MODEL_FILENAME = "sentiment_best_pipeline.pkl"
MODEL_PATH = Path(__file__).with_name(MODEL_FILENAME)


def load_model(model_path: Path | str = MODEL_PATH) -> Any:
    """Load the serialized best pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find sentiment model artifact at {path}. "
            "Please run the training notebook to generate it."
        )
    return joblib.load(path)


def _ensure_iterable(texts: Iterable[str] | str) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    return [str(t) for t in texts]


def clean_text(text: str) -> str:
    """Replicate the notebook preprocessing for inference-time parity."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for tok in text.split():
        if not tok.isalpha():
            continue
        if tok in ["dont", "didnt", "cant", "wont", "isnt", "arent", "wasnt", "werent"]:
            tokens.append("not")
            continue
        if tok in STOP_WORDS:
            continue
        tokens.append(STEMMER.stem(tok))

    return " ".join(tokens)


def _predict_with_prob(model: Any, cleaned_texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return labels and probabilities, emulating predict_proba if necessary."""
    labels = model.predict(cleaned_texts)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(cleaned_texts)
    else:
        decision_scores = model.decision_function(cleaned_texts)
        if decision_scores.ndim == 1:
            decision_scores = np.vstack([-decision_scores, decision_scores]).T
        probs = np.exp(decision_scores - decision_scores.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)

    return labels, probs


def predict_sentiment(texts: Iterable[str] | str, model: Any | None = None):
    """Return a list of {text, label, probability} dictionaries for the inputs."""
    if model is None:
        model = load_model()

    payload = _ensure_iterable(texts)
    clean_payload = [clean_text(t) for t in payload]
    labels, probs = _predict_with_prob(model, clean_payload)

    class_index = {cls: idx for idx, cls in enumerate(model.classes_)}
    results = []
    for original, label, prob_row in zip(payload, labels, probs):
        if hasattr(label, "item"):
            label = label.item()
        prob = float(prob_row[class_index[label]])
        results.append({"text": original, "label": label, "probability": prob})
    return results


def _demo():
    model = load_model()
    samples = [
        "Global markets rally as tech stocks surge ahead of earnings.",
        "Investors fear prolonged recession after weak manufacturing data.",
    ]
    for item in predict_sentiment(samples, model):
        print(f"Text: {item['text']}")
        print(f"Label: {item['label']} | Probability: {item['probability']:.4f}")
        print("-")


if __name__ == "__main__":
    _demo()
