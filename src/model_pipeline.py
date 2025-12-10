from transformers import pipeline
from typing import List, Dict

def load_summarizer(device: int = 0):
    """
    device:
      -1 = CPU
       0 = first GPU (on Vessl, this is usually what you want)
    """
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=device
    )
    return summarizer

def generate_summary_one(
    summarizer,
    text: str,
    min_len: int = 15,
    max_new_tokens: int = 80 # limit new tokens so model is not cut off
) -> str:
    result = summarizer(
        text,
        min_length=min_len,
        max_new_tokens=max_new_tokens,
        truncation=True
    )[0]["summary_text"]
    return result

def run_model_on_dataset(
    summarizer,
    examples: List[Dict],
    min_len: int = 15,
    max_new_tokens: int = 80
) -> List[str]:
    preds = []
    for ex in examples:
        doc = ex["document"]
        pred = generate_summary_one(summarizer, doc, min_len, max_new_tokens)
        preds.append(pred)
    return preds
