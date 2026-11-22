# Receipt / Invoice Information Extraction with Layout-Aware Transformers

**Type:** Multimodal (Image + Text)

## Problem Statement

Extract key fields (**company**, **date**, **address**, **total**) from scanned receipts using text and layout.

## Research Question

How much does layout information add over text alone? We answer this with a field-wise F1 comparison of three methods.

## Dataset

**SROIE** (ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction) [2]. We use two Hugging Face versions of the same receipts together:

| Version | Contents | Used for |
|---|---|---|
| [`darentang/sroie`](https://huggingface.co/datasets/darentang/sroie) | OCR words, word boxes (already scaled to 0–1000), one BIO tag per word | Training BERT / LayoutLM, regex input, main evaluation gold |
| [`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) | Receipt images, official ICDAR answers (company, date, address, total) | Figures, second evaluation against the official answers |

The two are matched on the receipt **key**, which is the image file name (e.g. `X51005230621`). `src/data.py` loads the tagged data (`load_sroie()`) and images (`load_images()`).

- **Train:** 626 receipts. **Test:** 347 receipts with word tags. `jsdnrs` has 14 extra test receipts without word tags, which we leave out.
- About 117 words per receipt on average. 84.7 % of words are `O` (not a field).
- Details and figures: `notebooks/01_data_exploration.ipynb`.

## Approach

| # | Method | Input | Notebook |
|---|--------|-------|----------|
| 1 | OCR + Regex baseline | Text (hand-written rules) | `notebooks/02_regex_baseline.ipynb` |
| 2 | BERT token classification | Text only | `notebooks/03_bert_text_only.ipynb` |
| 3 | LayoutLM | Text + 2D position | `notebooks/04_layoutlm.ipynb` |

All methods are scored with the same field-level evaluation (`src/evaluate.py`): exact match after normalisation, with precision, recall and F1 per field, on the 347 test receipts. We report two scores:

1. **Main:** against the gold string built from the tagged words. This measures whether a method picks the right words.
2. **Second:** against the official ICDAR answers. OCR misreads cap this score: even perfect tagging reaches only 82.8 % macro exact match (address 58.2 %). See `results/oracle_ceiling_vs_official.csv`.

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
  data.py                     # shared dataset loading (tagged words + images)
  evaluate.py                 # shared field-level evaluation
results/                      # predictions, metrics, figures
report/                       # project report (PDF)
requirements.txt
```

## How to Run

1. Open the notebooks in Google Colab (Runtime → Change runtime type → T4 GPU for notebooks 03 and 04).
2. Install dependencies: `pip install -r requirements.txt` (each notebook's first cell does this in Colab).
3. Run the notebooks in order, 01 → 05. In Colab, the setup cell clones this repo so `src/` can be imported.

## References

1. Xu et al., *LayoutLM: Pre-training of Text and Layout for Document Image Understanding*, KDD 2020. https://arxiv.org/abs/1912.13318
2. Huang et al., *ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction*, ICDAR 2019. https://arxiv.org/abs/2103.10213
