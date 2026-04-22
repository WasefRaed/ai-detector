from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import nltk
import spacy
import textstat
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="AI Text Detector API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model, tokenizer = None, None

# ── Linguistic Feature Extractor ─────────────────────────────────────────────
class LinguisticFeatureExtractor:
    TRANSITIONS = {
        "furthermore","moreover","additionally","consequently",
        "therefore","however","nevertheless","in conclusion",
        "in summary","to summarize","it is worth noting",
        "it should be noted","this demonstrates","this highlights"
    }

    def extract(self, text: str) -> dict:
        doc      = nlp(text)
        sentences = list(doc.sents)
        words     = [t.text for t in doc if not t.is_space and t.is_alpha]
        if not words or not sentences:
            return {}

        sent_lengths = [len(s.text.split()) for s in sentences]
        mean_len     = float(np.mean(sent_lengths))
        std_len      = float(np.std(sent_lengths))

        first_person = sum(
            1 for t in doc if t.text.lower() in ["i","me","my","myself","mine"]
        )
        text_lower   = text.lower()
        trans_count  = sum(text_lower.count(t) for t in self.TRANSITIONS)

        burstiness = (
            (std_len - mean_len) / (std_len + mean_len)
            if (std_len + mean_len) > 0 else 0
        )

        return {
            "avg_sentence_length":   round(mean_len, 2),
            "std_sentence_length":   round(std_len, 2),
            "type_token_ratio":      round(len(set(words)) / len(words), 3),
            "first_person_ratio":    round(first_person / len(words), 4),
            "transition_word_ratio": round(trans_count / len(words), 4),
            "named_entity_density":  round(len(doc.ents) / len(sentences), 3),
            "burstiness":            round(burstiness, 3),
            "flesch_reading_ease":   round(textstat.flesch_reading_ease(text), 2),
        }

    def generate_reasons(self, feats: dict, prediction: str) -> list:
        reasons = []
        if not feats:
            return reasons
    
        # ── AI signals — only fire when strongly indicative ──────────────────
        if feats.get("std_sentence_length", 99) < 2.8:
            reasons.append({
                "signal":      "repetitive_structure",
                "description": "Sentence lengths are unusually uniform — a strong indicator of AI generation.",
                "severity":    "high"
            })
    
        if feats.get("transition_word_ratio", 0) > 0.02:
            reasons.append({
                "signal":      "overuse_of_transitions",
                "description": "High use of transition words like 'furthermore' and 'moreover' — a typical AI writing pattern.",
                "severity":    "medium"
            })
    
        if feats.get("burstiness", 0) < -0.4:
            reasons.append({
                "signal":      "low_burstiness",
                "description": "Writing rhythm is too consistent. Human writing naturally varies in pace and sentence length.",
                "severity":    "medium"
            })
    
        if feats.get("first_person_ratio", 1) < 0.001:
            reasons.append({
                "signal":      "lack_of_personal_detail",
                "description": "Almost no first-person pronouns detected — AI tends to avoid personal voice.",
                "severity":    "medium"
            })
    
        if feats.get("type_token_ratio", 0) > 0.92:
            reasons.append({
                "signal":      "high_vocabulary_diversity",
                "description": "Unusually high vocabulary variety — AI models often over-diversify word choice.",
                "severity":    "medium"
            })
    
        # ── Human signals — only fire when strongly indicative ───────────────
        if feats.get("first_person_ratio", 0) > 0.04:
            reasons.append({
                "signal":      "personal_voice_present",
                "description": "Strong use of first-person voice suggests authentic human authorship.",
                "severity":    "positive"
            })
    
        if feats.get("named_entity_density", 0) > 1.8:
            reasons.append({
                "signal":      "specific_references",
                "description": "High density of named entities suggests genuine human knowledge and experience.",
                "severity":    "positive"
            })
    
        if feats.get("burstiness", 0) > 0.2:
            reasons.append({
                "signal":      "natural_rhythm",
                "description": "Writing shows natural variation in pace — consistent with human authorship.",
                "severity":    "positive"
            })
    
        if feats.get("std_sentence_length", 0) > 12:
            reasons.append({
                "signal":      "varied_sentence_structure",
                "description": "High variation in sentence length is a natural pattern of human writing.",
                "severity":    "positive"
            })
    
        # ── Fallback — if nothing fired, explain the verdict simply ──────────
        if not reasons:
            if prediction == "AI":
                reasons.append({
                    "signal":      "pattern_match",
                    "description": "The model detected subtle statistical patterns consistent with AI-generated text.",
                    "severity":    "medium"
                })
            else:
                reasons.append({
                    "signal":      "natural_writing_patterns",
                    "description": "The text shows natural human writing patterns without strong AI indicators.",
                    "severity":    "positive"
                })
    
        return reasons

extractor = LinguisticFeatureExtractor()

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
    model     = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=3)   
    model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
    model.eval()
    print("✅ Model loaded! (3-class: Human / Hybrid / AI)")

# ── Routes ────────────────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    text = request.text.strip()
    if len(text) < 50:
        raise HTTPException(400, "Text too short — minimum 50 characters")

    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=256, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]

    human_prob  = float(probs[0])
    hybrid_prob = float(probs[1])
    ai_prob     = float(probs[2])

    scores     = {"Human": human_prob, "Hybrid": hybrid_prob, "AI": ai_prob}
    prediction = max(scores, key=scores.get)

    feats   = extractor.extract(text)
    reasons = extractor.generate_reasons(feats, prediction)

    return {
        "prediction":  prediction,
        "confidence":  round(max(human_prob, hybrid_prob, ai_prob), 4),
        "probabilities": {
            "human":  round(human_prob,  4),
            "hybrid": round(hybrid_prob, 4),
            "ai":     round(ai_prob,     4),
        },
        "reasons":             reasons,
        "linguistic_features": feats,
    }

@app.get("/health")
async def health():
    return {"status": "ok"}