import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading all three classes...")
train_df  = pd.read_csv("data/processed/train.csv")
val_df    = pd.read_csv("data/processed/val.csv")
test_df   = pd.read_csv("data/processed/test.csv")
hybrid_df = pd.read_csv("data/raw/hybrid_texts.csv")

# Remap old labels: 0=Human, 2=AI → 0=Human, 2=AI (keep as is)
# New label: 1=Hybrid
full_df = pd.concat([train_df, val_df, test_df])
human_df = full_df[full_df["label"] == 0]
ai_df    = full_df[full_df["label"] == 2].copy()
ai_df["label"] = 2  # keep as 2

# Balance all three classes
min_size = min(len(human_df), len(ai_df), len(hybrid_df))
print(f"Balancing to {min_size} samples per class...")

human_df  = human_df.sample(min_size, random_state=42)
ai_df     = ai_df.sample(min_size, random_state=42)
hybrid_df = hybrid_df.sample(min_size, random_state=42)

# Combine and shuffle
df = pd.concat([human_df, ai_df, hybrid_df]).sample(frac=1, random_state=42)
df = df[["text", "label"]].dropna()

print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"  0 = Human | 1 = Hybrid | 2 = AI")

# Split
train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val, test   = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

train.to_csv("data/processed/train_v2.csv", index=False)
val.to_csv("data/processed/val_v2.csv",     index=False)
test.to_csv("data/processed/test_v2.csv",   index=False)

print(f"\n✅ Done!")
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")