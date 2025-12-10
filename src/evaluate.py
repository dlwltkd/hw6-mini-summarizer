# src/evaluate.py
from typing import List, Dict
from baseline import lead1_summary
from rouge_score import rouge_scorer


def run_baseline_on_dataset(examples: List[Dict]) -> List[str]:
    predictions = []
    for ex in examples:
        doc = ex["document"]
        pred = lead1_summary(doc)
        predictions.append(pred)
    return predictions

def compute_rouge_scores(
    references: List[str],
    predictions: List[str]
) -> List[Dict]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = []
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        scores.append(score)
    return scores

def average_rouge(scores: List[Dict]) -> Dict[str, float]:
    n = len(scores)
    avg = {
        "rouge1": 0.0,
        "rougeL": 0.0,
    }
    for s in scores:
        avg["rouge1"] += s["rouge1"].fmeasure
        avg["rougeL"] += s["rougeL"].fmeasure
    avg["rouge1"] /= n
    avg["rougeL"] /= n
    return avg