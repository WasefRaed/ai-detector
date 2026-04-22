# src/data/create_hybrid.py
# Strategy: Take human text → ask GPT to "improve" it → result is Hybrid

def create_hybrid(human_text: str, client) -> str:
    """
    Ask GPT to lightly edit a human text.
    The output retains human structure but has AI polish = Hybrid.
    """
    prompt = f"""
    Lightly edit the following text to improve clarity and flow.
    Keep the original ideas, voice, and structure.
    Only fix grammar and awkward phrasing.
    
    Text: {human_text}
    
    Edited version:
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3  # Low temp = subtle edits
    )
    return response.choices[0].message.content

# Process in batches
human_df = pd.read_csv("data/raw/human_texts.csv")
hybrid_results = []

for _, row in human_df.iterrows():
    hybrid_text = create_hybrid(row["text"], client)
    hybrid_results.append({
        "text": hybrid_text,
        "original_human": row["text"],
        "label": "HYBRID"
    })

hybrid_df = pd.DataFrame(hybrid_results)
hybrid_df.to_csv("data/raw/hybrid_texts.csv", index=False)