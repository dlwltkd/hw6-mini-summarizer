# Mini Summarizer (HW6)

A small course project that experiments with text summarization on a tiny slice of the CNN/DailyMail dataset. It includes a baseline (lead-1) and a transformer-based pipeline built with Hugging Face.

## Project Structure
- `run_experiment.ipynb`: end-to-end notebook to build the dataset, run models, and visualize results.
- `src/build_dataset.py`: sample and clean a subset of CNN/DailyMail into `data/summarization_data.json`.
- `src/baseline.py`: lead-1 sentence baseline summarizer.
- `src/model_pipeline.py`: wrapper around `sshleifer/distilbart-cnn-12-6` via `transformers.pipeline`.
- `src/evaluate.py`: run summarizers and compute ROUGE-1/L.
- `data/summarization_data.json`: prebuilt small dataset (train + test).
- `report.pdf`: written project report with methodology and findings.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
From the project root:

### Quickest: run the notebook
- Open `run_experiment.ipynb` and execute all cells. It walks through building the dataset, running the lead-1 baseline, running the transformer summarizer, and showing ROUGE scores/plots in one place.

### CLI equivalents (optional)
1) Build (or rebuild) the tiny dataset:
```bash
python src/build_dataset.py
```

2) Run the lead-1 baseline on the dataset and compute ROUGE:
```bash
python - <<'PY'
import json
from src import evaluate, baseline

with open('data/summarization_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

preds = evaluate.run_baseline_on_dataset(data)
refs = [ex['summary'] for ex in data]
scores = evaluate.compute_rouge_scores(refs, preds)
print(evaluate.average_rouge(scores))
PY
```

3) Run the transformer model (CPU by default; set `device=0` for GPU if available):
```bash
python - <<'PY'
import json
from src import model_pipeline, evaluate

with open('data/summarization_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

summarizer = model_pipeline.load_summarizer(device=-1)
preds = model_pipeline.run_model_on_dataset(summarizer, data)
refs = [ex['summary'] for ex in data]
scores = evaluate.compute_rouge_scores(refs, preds)
print(evaluate.average_rouge(scores))
PY
```

## Notes
- The dataset builder uses `random.seed(42)` for repeatability and samples 200 train / 50 test examples.
- `max_new_tokens` is limited in `model_pipeline` to avoid overly long outputs.
- ROUGE computation uses stemming with `rouge_score`.
