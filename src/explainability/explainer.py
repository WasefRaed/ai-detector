# src/explainability/explainer.py
import shap
import numpy as np
import torch
from transformers import pipeline

class TextExplainer:
    """
    Explains WHY the model made its decision.
    Highlights suspicious phrases and gives human-readable reasons.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.label_names = ["Human", "Hybrid", "AI"]

        # SHAP explainer using model as a text classifier
        self.explainer = shap.Explainer(self._predict_proba, tokenizer)

    def _predict_proba(self, texts):
        """Wrapper for SHAP — returns probability array"""
        probs = []
        for text in texts:
            enc = self.tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.model(**enc)
            prob = torch.softmax(logits, dim=1).numpy()[0]
            probs.append(prob)
        return np.array(probs)

    def explain(self, text: str, linguistic_features: dict,
                perplexity_data: dict) -> dict:
        """Full explanation pipeline"""

        # 1. Get prediction
        probs = self._predict_proba([text])[0]
        predicted_class = np.argmax(probs)

        # 2. Get SHAP token-level attributions
        shap_values = self.explainer([text])
        token_weights = shap_values.values[0, :, predicted_class]
        tokens = shap_values.data[0]

        # 3. Find most suspicious spans (high SHAP weight tokens)
        token_data = sorted(
            zip(tokens, token_weights),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        suspicious_tokens = [(t, float(w)) for t, w in token_data[:10]
                             if w > 0]  # Positive = pushed toward AI

        # 4. Generate human-readable reasoning
        reasons = self._generate_reasons(
            predicted_class, linguistic_features, perplexity_data, probs
        )

        # 5. Sentence-level highlights from perplexity
        sentence_highlights = self._highlight_sentences(
            perplexity_data["sentence_scores"]
        )

        return {
            "prediction":           self.label_names[predicted_class],
            "confidence":           float(probs[predicted_class]),
            "probabilities": {
                "human":  float(probs[0]),
                "hybrid": float(probs[1]),
                "ai":     float(probs[2])
            },
            "suspicious_phrases":   suspicious_tokens,
            "sentence_highlights":  sentence_highlights,
            "reasons":              reasons
        }

    def _generate_reasons(self, pred_class, ling_feats,
                          perp_data, probs) -> list:
        """
        Rule-based reasoning layer.
        Maps linguistic signals to readable explanations.
        """
        reasons = []

        # --- AI signals ---
        if ling_feats["std_sentence_length"] < 8:
            reasons.append({
                "signal":      "repetitive_structure",
                "description": "Sentence lengths are unusually uniform — "
                               "a strong indicator of AI generation.",
                "severity":    "high"
            })

        if ling_feats["transition_word_ratio"] > 0.02:
            reasons.append({
                "signal":      "overuse_of_transitions",
                "description": "High density of transition words like "
                               "'furthermore' and 'moreover' — typical AI pattern.",
                "severity":    "medium"
            })

        if perp_data["perplexity"] < 30:
            reasons.append({
                "signal":      "unnatural_fluency",
                "description": f"Text perplexity is very low ({perp_data['perplexity']:.1f}), "
                               f"meaning it is highly predictable — common in AI output.",
                "severity":    "high"
            })

        if ling_feats["first_person_ratio"] < 0.005:
            reasons.append({
                "signal":      "lack_of_personal_detail",
                "description": "Almost no first-person pronouns — "
                               "AI avoids personal voice and lived experience.",
                "severity":    "medium"
            })

        if ling_feats["burstiness"] < -0.2:
            reasons.append({
                "signal":      "low_burstiness",
                "description": "Writing rhythm is too consistent. "
                               "Human writing naturally varies in pace and length.",
                "severity":    "medium"
            })

        # --- Human signals ---
        if ling_feats["first_person_ratio"] > 0.03:
            reasons.append({
                "signal":      "personal_voice_present",
                "description": "Strong use of first-person voice suggests "
                               "authentic human authorship.",
                "severity":    "positive"
            })

        if ling_feats["named_entity_density"] > 1.5:
            reasons.append({
                "signal":      "specific_references",
                "description": "High density of named entities (people, places, events) "
                               "suggests genuine human knowledge and experience.",
                "severity":    "positive"
            })

        return reasons

    def _highlight_sentences(self, sentence_scores: list) -> list:
        """Flag sentences with suspiciously low perplexity"""
        if not sentence_scores:
            return []
        all_perp = [s["perplexity"] for s in sentence_scores]
        threshold = np.percentile(all_perp, 25)  # Bottom 25% = suspicious
        return [
            {**s, "flagged": s["perplexity"] < threshold}
            for s in sentence_scores
        ]