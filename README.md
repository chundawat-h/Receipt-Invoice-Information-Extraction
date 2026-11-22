# Receipt / Invoice Information Extraction with Layout-Aware Transformers

**Type:** Multimodal (Image + Text)

## Problem Statement

Extract key fields (**company**, **date**, **address**, **total**) from scanned receipts using text and layout.

## Research Question

How much does layout information add over text alone? We answer this with a field-wise F1 comparison of three methods.

## Dataset

**SROIE** (ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction), loaded from Hugging Face as [`darentang/sroie`](https://huggingface.co/datasets/darentang/sroie). It includes OCR words, word bounding boxes and BIO labels for each receipt.

## Approach

| # | Method | Input | Notebook |
|---|--------|-------|----------|
| 1 | OCR + Regex baseline | Text (hand-written rules) | `notebooks/02_regex_baseline.ipynb` |
| 2 | BERT token classification | Text only | `notebooks/03_bert_text_only.ipynb` |
| 3 | LayoutLM | Text + 2D position | `notebooks/04_layoutlm.ipynb` |

All methods are scored with the same field-level evaluation (`src/evaluate.py`): exact match after normalisation, with precision, recall and F1 per field.

## Results

_To be added._

## Project Structure

```
notebooks/
  01_data_exploration.ipynb   # dataset stats and visualisation
  02_regex_baseline.ipynb     # Method 1
  03_bert_text_only.ipynb     # Method 2
  04_layoutlm.ipynb           # Method 3
  05_results_analysis.ipynb   # comparison and field-wise F1 analysis
src/
  evaluate.py                 # shared field-level evaluation
results/                      # predictions, metrics, figures
report/                       # project report (PDF)
requirements.txt
```

## How to Run

1. Open the notebooks in Google Colab (Runtime → Change runtime type → T4 GPU for notebooks 03 and 04).
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks in order, 01 → 05.

## References

1. Xu et al., *LayoutLM: Pre-training of Text and Layout for Document Image Understanding*, KDD 2020. https://arxiv.org/abs/1912.13318
