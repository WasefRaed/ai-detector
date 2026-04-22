# src/data/collect_human.py
from datasets import load_dataset

# Option A: Writing Prompts (Reddit creative writing)
dataset = load_dataset("reddit_writing_prompts")

# Option B: News articles
dataset = load_dataset("cc_news")

# Option C: Academic essays
dataset = load_dataset("persuade_corpus")  # Student essays

# Option D: Human ChatGPT Comparison Corpus
dataset = load_dataset("Hello-SimpleAI/HC3", "all")
# This gives you human Q&A answers vs ChatGPT answers — perfect

# Save
dataset.to_csv("data/raw/human_texts.csv")