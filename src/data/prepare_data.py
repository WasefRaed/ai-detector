from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

print("Downloading dataset...")

# Load RAID dataset - modern AI detection benchmark
dataset = load_dataset("liamdugan/raid", split="train")

rows = []
for item in dataset:
    text = item.get("generation") or item.get("text") or ""
    label_raw = item.get("model") or ""

    if len(text.strip()) < 100:
        continue

    # Human = label 0, AI = label 2
    if label_raw == "human":
        rows.append({"text": text.strip()[:2000], "label": 0})
    else:
        rows.append({"text": text.strip()[:2000], "label": 2})

df = pd.DataFrame(rows)
print(f"Human: {len(df[df.label==0])} | AI: {len(df[df.label==2])}")

# Balance
min_size = min(len(df[df.label==0]), len(df[df.label==2]))
df = pd.concat([
    df[df.label==0].sample(min_size, random_state=42),
    df[df.label==2].sample(min_size, random_state=42)
]).sample(frac=1, random_state=42)

# Split
train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val, test   = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv",     index=False)
test.to_csv("data/processed/test.csv",   index=False)

print(f"✅ Done! Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")