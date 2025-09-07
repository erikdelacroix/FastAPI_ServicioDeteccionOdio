# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline



# === Config ===
MODEL_DIR = "api/models_multiclase_final"
LABELS = {
    0: "no_discriminatorio",
    1: "racismo_xenofobia",
    2: "machismo_sexismo",
    3: "homofobia_LGTBIQ",
}
LABEL2ID = {v: k for k, v in LABELS.items()}

# Carga única en el arranque
device = 0 if torch.cuda.is_available() else -1

# Carga del modelo/tokenizer
model_id = "erikcruzuc3m/model-multiclase-final"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    return_all_scores=True,
    truncation=True,
    max_length=256,
)

# Esquemas
class PredictRequest(BaseModel):
    text: str
    recall_boost: bool = False
    min_other: float = 0.35
    max_top0: float = 0.55

class PredictBatchRequest(BaseModel):
    texts: conlist(str, min_length=1)
    recall_boost: bool = False
    min_other: float = 0.35
    max_top0: float = 0.55

# Utilidades
def postprocess_scores(score_items: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for it in score_items:
        lab = it["label"]
        if lab.startswith("LABEL_"):
            idx = int(lab.split("_")[-1])
        else:
            try:
                idx = int(lab)
            except:
                idx = LABEL2ID.get(lab, 0)
        out[LABELS.get(idx, str(idx))] = float(it["score"])
    return out

def recall_boost_fix(probs: Dict[str, float], min_other: float, max_top0: float) -> str:
    top_label = max(probs, key=probs.get)
    if top_label != LABELS[0]:
        return top_label
    top0 = probs[LABELS[0]]
    other_sum = probs[LABELS[1]] + probs[LABELS[2]] + probs[LABELS[3]]
    if (top0 <= max_top0) and (other_sum >= min_other):
        pick = max([LABELS[1], LABELS[2], LABELS[3]], key=lambda k: probs[k])
        return pick
    return top_label

# FastAPI app
app = FastAPI(title="Deteccion de discurso de odio - API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if device >= 0 else "cpu"}

@app.post("/predict")
def predict(req: PredictRequest):
    out = clf(req.text)[0]
    probs = postprocess_scores(out)
    label = max(probs, key=probs.get)
    if req.recall_boost:
        label = recall_boost_fix(probs, req.min_other, req.max_top0)
    return {"label": label, "probs": probs}

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    results = []
    raw = clf(req.texts)
    for scores in raw:
        probs = postprocess_scores(scores)
        lab = max(probs, key=probs.get)
        if req.recall_boost:
            lab = recall_boost_fix(probs, req.min_other, req.max_top0)
        results.append({"label": lab, "probs": probs})
    return {"results": results, "n": len(results)}
