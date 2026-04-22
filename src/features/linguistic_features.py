# src/features/linguistic_features.py
import nltk
import spacy
import numpy as np
import textstat
from collections import Counter

nlp = spacy.load("en_core_web_sm")
nltk.download(["punkt", "averaged_perceptron_tagger", "stopwords"])

class LinguisticFeatureExtractor:
    """
    Extracts features that distinguish AI from human writing.
    These directly feed the explainability layer.
    """

    def extract(self, text: str) -> dict:
        doc = nlp(text)
        sentences = list(doc.sents)
        words = [t.text for t in doc if not t.is_space]

        return {
            # --- Fluency features (AI tends to be unnaturally smooth) ---
            "flesch_reading_ease":    textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade":   textstat.flesch_kincaid_grade(text),
            "gunning_fog":            textstat.gunning_fog(text),

            # --- Sentence structure (AI has low variance = repetitive) ---
            "avg_sentence_length":    np.mean([len(s) for s in sentences]),
            "std_sentence_length":    np.std([len(s) for s in sentences]),   # LOW = AI red flag
            "sentence_count":         len(sentences),

            # --- Vocabulary richness (AI repeats patterns) ---
            "type_token_ratio":       len(set(words)) / len(words) if words else 0,
            "avg_word_length":        np.mean([len(w) for w in words]) if words else 0,

            # --- Punctuation patterns ---
            "comma_ratio":            text.count(",") / len(words) if words else 0,
            "exclamation_ratio":      text.count("!") / len(sentences) if sentences else 0,
            "question_ratio":         text.count("?") / len(sentences) if sentences else 0,

            # --- Personal voice (AI lacks first-person depth) ---
            "first_person_ratio":     sum(1 for t in doc if t.text.lower() in
                                      ["i","me","my","myself","mine"]) / len(words) if words else 0,

            # --- Burstiness (human writing is bursty, AI is flat) ---
            "burstiness":             self._burstiness(sentences),

            # --- Transition words (AI overuses them) ---
            "transition_word_ratio":  self._transition_ratio(text, words),

            # --- Named entity density (humans reference real specifics) ---
            "named_entity_density":   len(doc.ents) / len(sentences) if sentences else 0,
        }

    def _burstiness(self, sentences) -> float:
        """
        Measures variation in sentence length.
        Humans write in bursts (short then long sentences).
        AI writes uniformly — low burstiness is an AI signal.
        """
        lengths = [len(s) for s in sentences]
        if len(lengths) < 2:
            return 0
        mean, std = np.mean(lengths), np.std(lengths)
        return (std - mean) / (std + mean) if (std + mean) > 0 else 0

    def _transition_ratio(self, text: str, words: list) -> float:
        """AI overuses transitions: 'Furthermore', 'Moreover', 'In conclusion'"""
        transitions = {
            "furthermore","moreover","additionally","consequently",
            "therefore","however","nevertheless","in conclusion",
            "in summary","to summarize","it is worth noting"
        }
        text_lower = text.lower()
        count = sum(text_lower.count(t) for t in transitions)
        return count / len(words) if words else 0