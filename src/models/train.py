import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import wandb
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):  
        self.texts    = texts
        self.labels   = labels
        self.tokenizer = tokenizer
        self.max_len  = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Training ──────────────────────────────────────────────────────────────────
def train_model(config):
    wandb.init(project="ai-text-detector", config=config)

    train_df = pd.read_csv("data/processed/train_v2.csv")
    val_df   = pd.read_csv("data/processed/val_v2.csv")

    # Remap labels: 
    train_df["label"] = train_df["label"].map({0: 0, 1: 1, 2: 2})
    val_df["label"]   = val_df["label"].map({0: 0, 1: 1, 2: 2})

    # Use smaller subset to fit in 2GB VRAM
    train_df = train_df.sample(min(60000, len(train_df)), random_state=42)
    val_df   = val_df.sample(min(6000,  len(val_df)),   random_state=42)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # DistilBERT — 40% smaller than BERT, fits in 2GB GPU
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForSequenceClassification.from_pretrained(
                     MODEL_NAME, num_labels=3)
    model      = model.to(config["device"])

    train_ds = TextDataset(train_df["text"].tolist(),
                           train_df["label"].tolist(), tokenizer)
    val_ds   = TextDataset(val_df["text"].tolist(),
                           val_df["label"].tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(config["device"])
            attn_mask = batch["attention_mask"].to(config["device"])
            labels    = batch["label"].to(config["device"])

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            loss    = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            correct    += (outputs.logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            if i % 100 == 0:
                print(f"  Epoch {epoch+1} | Step {i}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f} "
                      f"| Acc: {correct/max(total,1):.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config["device"])
                attn_mask = batch["attention_mask"].to(config["device"])
                labels    = batch["label"].to(config["device"])
                outputs   = model(input_ids=input_ids, attention_mask=attn_mask)
                val_preds.extend(outputs.logits.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        print(f"\nEpoch {epoch+1} complete | Val Acc: {val_acc:.4f}")
        print(classification_report(val_labels, val_preds,
              target_names=["Human", "Hybrid", "AI"]))

        wandb.log({"val_accuracy": val_acc, "train_loss": train_loss})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pt")
            # Save tokenizer too (needed for API)
            tokenizer.save_pretrained("models/tokenizer")
            print("✅ New best model saved!")

    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.4f}")

# ── Run ───────────────────────────────────────────────────────────────────────
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 5
}
print(f"Using device: {config['device']}")
train_model(config)