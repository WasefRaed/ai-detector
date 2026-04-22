# src/features/perplexity_features.py
# Perplexity measures how "surprising" text is to a language model.
# AI text has LOW perplexity (predictable). Human text has HIGH perplexity.

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class PerplexityCalculator:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model.eval()

    def calculate(self, text: str) -> dict:
        encodings = self.tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss

        perplexity = torch.exp(loss).item()

        # Also calculate per-sentence perplexity for span highlighting
        sentences = text.split(".")
        sentence_scores = []
        for sent in sentences:
            if len(sent.strip()) > 10:
                enc = self.tokenizer(sent, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    out = self.model(**enc, labels=enc["input_ids"])
                sentence_scores.append({
                    "sentence": sent.strip(),
                    "perplexity": torch.exp(out.loss).item()
                })

        return {
            "perplexity": perplexity,
            "sentence_scores": sentence_scores  # Used for phrase highlighting
        }