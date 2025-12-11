from datasets import load_dataset
import json
import random

def clean_text(text: str) -> str:
    return " ".join(text.strip().split())

def build_dataset():
    # 1. Load dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    #  Shuffle with fixed seed for reproducibility
    train_indices = list(range(len(train_ds)))
    test_indices = list(range(len(test_ds)))

    random.seed(42)
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    # Choose subset sizes
    N_train = 200
    N_test = 50

    train_indices = train_indices[:N_train]
    test_indices = test_indices[:N_test]

    # Build custom list of dicts
    examples = []
    for idx in train_indices:
        article = clean_text(train_ds[idx]["article"])
        summary = clean_text(train_ds[idx]["highlights"])
        examples.append({"split": "train", "document": article, "summary": summary})

    for idx in test_indices:
        article = clean_text(test_ds[idx]["article"])
        summary = clean_text(test_ds[idx]["highlights"])
        examples.append({"split": "test", "document": article, "summary": summary})

    output_path = "data/summarization_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(examples)} examples to {output_path}")
