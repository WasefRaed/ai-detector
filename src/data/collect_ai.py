# src/data/collect_ai.py
import openai
import pandas as pd

def generate_ai_texts(prompts: list, model="gpt-3.5-turbo"):
    results = []
    for prompt in prompts:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        results.append({
            "text": response.choices[0].message.content,
            "label": "AI",
            "model": model,
            "prompt": prompt
        })
    return pd.DataFrame(results)

# Use prompts from your human dataset so topics match
prompts = pd.read_csv("data/raw/human_texts.csv")["prompt"].tolist()
ai_df = generate_ai_texts(prompts[:1000])
ai_df.to_csv("data/raw/ai_texts.csv", index=False)