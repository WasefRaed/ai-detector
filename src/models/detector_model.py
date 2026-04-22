# src/models/detector_model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class AITextDetector(nn.Module):
    """
    Hybrid model: RoBERTa contextual features + handcrafted linguistic features
    This is more explainable than a pure transformer.
    """
    def __init__(self, transformer_name="roberta-base", num_linguistic_features=15):
        super().__init__()

        # Transformer backbone
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size  # 768 for roberta-base

        # Fusion layer: combine transformer + linguistic features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + num_linguistic_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)   # 3 classes: Human, Hybrid, AI
        )

    def forward(self, input_ids, attention_mask, linguistic_features=None):
        # Get transformer [CLS] representation
        outputs = self.transformer(input_ids=input_ids,
                                   attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Concatenate with linguistic features
        if linguistic_features is not None:
            combined = torch.cat([cls_output, linguistic_features], dim=1)
        else:
            combined = cls_output

        return self.fusion(combined)