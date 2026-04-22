import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split

random.seed(42)

# ── Hybrid transformation functions ──────────────────────────────────────────

TRANSITIONS = [
    "Furthermore, ", "Moreover, ", "Additionally, ",
    "In fact, ", "Notably, ", "It is worth mentioning that ",
]

FILLER_ENDINGS = [
    " This is an important consideration.",
    " This highlights a key aspect of the topic.",
    " These factors are worth keeping in mind.",
]

def inject_transitions(text: str) -> str:
    """Insert AI-style transition words between sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < 3:
        return text
    # Inject a transition before 1-2 random sentences
    num_injections = min(2, len(sentences) - 1)
    indices = random.sample(range(1, len(sentences)), num_injections)
    for idx in indices:
        transition = random.choice(TRANSITIONS)
        first_word = sentences[idx].split()[0] if sentences[idx].split() else ""
        if first_word and first_word[0].isupper():
            sentences[idx] = transition + sentences[idx][0].lower() + sentences[idx][1:]
    return " ".join(sentences)

def normalize_sentences(text: str) -> str:
    """Make sentence lengths more uniform — an AI pattern."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    normalized = []
    for sent in sentences:
        words = sent.split()
        # Trim very long sentences slightly
        if len(words) > 30:
            sent = " ".join(words[:28]) + "."
        # Pad very short sentences with a filler
        elif len(words) < 6 and normalized:
            sent = sent.rstrip(".!?") + random.choice(FILLER_ENDINGS)
        normalized.append(sent)
    return " ".join(normalized)

def remove_casual_language(text: str) -> str:
    """Replace informal phrases with formal equivalents."""
    replacements = {
        r"\bI've\b":     "I have",
        r"\bI'm\b":      "I am",
        r"\bdon't\b":    "do not",
        r"\bcan't\b":    "cannot",
        r"\bwon't\b":    "will not",
        r"\bdidn't\b":   "did not",
        r"\bit's\b":     "it is",
        r"\bthat's\b":   "that is",
        r"\bthey're\b":  "they are",
        r"\bwe're\b":    "we are",
        r"\bHonestly,?\b": "In reality,",
        r"\bBasically,?\b": "Essentially,",
        r"\bActually,?\b":  "In actuality,",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def create_hybrid(text: str) -> str:
    """
    Apply all three transformations to simulate
    a human text that has been AI-polished.
    """
    text = remove_casual_language(text)
    text = inject_transitions(text)
    text = normalize_sentences(text)
    return text

# ── Load & process ────────────────────────────────────────────────────────────
print("Loading human texts...")
train_df = pd.read_csv("data/processed/train.csv")
human_df = train_df[train_df["label"] == 0].sample(8000, random_state=42)

print("Generating hybrid texts...")
hybrids = []
for i, (_, row) in enumerate(human_df.iterrows()):
    hybrid_text = create_hybrid(row["text"])
    # Only keep it if it actually changed
    if hybrid_text.strip() != row["text"].strip():
        hybrids.append({"text": hybrid_text.strip()[:2000], "label": 1})
    if i % 1000 == 0:
        print(f"  {i}/{len(human_df)} processed...")

hybrid_df = pd.DataFrame(hybrids)
hybrid_df.to_csv("data/raw/hybrid_texts.csv", index=False)
print(f"✅ Done! Generated {len(hybrid_df)} hybrid samples.")
print(hybrid_df.head(2))