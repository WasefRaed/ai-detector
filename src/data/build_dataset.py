# src/data/build_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Load all three
human = pd.read_csv("data/raw/human_texts.csv")[["text"]].assign(label=0)    # 0 = Human
ai    = pd.read_csv("data/raw/ai_texts.csv")[["text"]].assign(label=2)       # 2 = AI
hybrid= pd.read_csv("data/raw/hybrid_texts.csv")[["text"]].assign(label=1)   # 1 = Hybrid

# Balance classes (equal samples from each)
min_size = min(len(human), len(ai), len(hybrid))
human  = human.sample(min_size, random_state=42)
ai     = ai.sample(min_size, random_state=42)
hybrid = hybrid.sample(min_size, random_state=42)

# Combine
df = pd.concat([human, ai, hybrid]).sample(frac=1, random_state=42)

# Clean text
df["text"] = df["text"].str.strip().str[:2000]  # Limit length
df = df[df["text"].str.len() > 100]             # Remove very short texts
df = df.drop_duplicates(subset="text")

# Split
train, temp = train_test_split(df, test_size=0.2, stratify=df["label"])
val, test   = train_test_split(temp, test_size=0.5, stratify=temp["label"])

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
# Label distribution
print(df["label"].value_counts())