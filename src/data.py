"""Shared SROIE loading for all notebooks.

Two Hugging Face versions of SROIE (ICDAR 2019) are used together
(see notebooks/01_data_exploration.ipynb and issue #4):

* ``darentang/sroie``: one BIO tag per word and word boxes already scaled to
  0-1000. Used for training (BERT, LayoutLM), the regex baseline input, and the
  main "tagged-word" gold for evaluation.
* ``jsdnrs/ICDAR2019-SROIE``: receipt images and the official ICDAR answers.
  Official answers are loaded in ``src/evaluate.py`` (``get_official_gold``).

Receipts are matched on ``key``, the image file name without extension
(e.g. ``"X51005230621"``). Prediction files use this key.

Usage (in a notebook, with the repo root on ``sys.path``)::

    from src.data import load_sroie, LABELS, ID2LABEL, LABEL2ID

    ds = load_sroie()          # DatasetDict: train (626) / test (347)
    row = ds["test"][0]        # id, words, bboxes, ner_tags, image_path, key
"""

from pathlib import Path

from datasets import load_dataset

LABELS = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE", "I-DATE",
    "B-ADDRESS", "I-ADDRESS",
    "B-TOTAL", "I-TOTAL",
]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {label: i for i, label in ID2LABEL.items()}

TAGGED = "darentang/sroie"
OFFICIAL = "jsdnrs/ICDAR2019-SROIE"


def receipt_key(image_path):
    """Image path -> receipt key, e.g. '.../X51005230621.jpg' -> 'X51005230621'."""
    return Path(image_path).stem


def load_sroie():
    """Load the word-tagged SROIE splits (train 626 / test 347) with a ``key`` column.

    Uses the Hugging Face Parquet conversion of ``darentang/sroie``: same rows as
    the original loader, but no Google Drive download and no remote code, and it
    works with any ``datasets`` version. ``bboxes`` are already 0-1000.
    """
    ds = load_dataset(TAGGED, revision="refs/convert/parquet")
    names = ds["train"].features["ner_tags"].feature.names
    if names != LABELS:
        raise ValueError(f"unexpected label set: {names}")
    return ds.map(lambda row: {"key": receipt_key(row["image_path"])})


def load_images(split, keys):
    """Return ``{key: PIL.Image}`` for the given receipt keys.

    Images come from ``jsdnrs/ICDAR2019-SROIE``. The split is streamed and the
    loop stops once all keys are found, so the 500 MB dataset is not downloaded.
    """
    keys, found = set(keys), {}
    rows = load_dataset(OFFICIAL, split=split, streaming=True, columns=["key", "image"])
    for row in rows:
        if row["key"] in keys:
            found[row["key"]] = row["image"].convert("RGB")
            if len(found) == len(keys):
                break
    missing = keys - set(found)
    if missing:
        raise KeyError(f"no image for keys: {sorted(missing)[:5]}")
    return found
