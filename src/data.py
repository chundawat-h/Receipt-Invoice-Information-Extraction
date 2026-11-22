"""Load the two Hugging Face versions of SROIE (ICDAR 2019) and join them.

1. ``darentang/sroie``: OCR words, word boxes (already scaled to 0-1000) and
   word-level BIO tags. Used for training and as the main gold standard.
   Its repo only has a loading script, which newer ``datasets`` versions refuse
   to run, so we load Hugging Face's auto-converted Parquet copy
   (``revision="refs/convert/parquet"``). This copy has no images.
2. ``jsdnrs/ICDAR2019-SROIE``: the receipt **images** and the **official**
   ICDAR answers (``entities``). About 500 MB, so it is only loaded when needed.

Both are joined on the receipt key: the image file name without extension,
e.g. ``X51005230621``. All 626 train and 347 test receipts of darentang are in
jsdnrs (which has 361 test receipts; we use the 347 shared ones).
"""

from pathlib import Path

from datasets import load_dataset

TAGGED_DATASET = "darentang/sroie"
OFFICIAL_DATASET = "jsdnrs/ICDAR2019-SROIE"

LABELS = ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE",
          "B-ADDRESS", "I-ADDRESS", "B-TOTAL", "I-TOTAL"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {label: i for i, label in ID2LABEL.items()}


def receipt_key(image_path):
    """'.../sroie/test/images/X00016469670.jpg' -> 'X00016469670'."""
    return Path(image_path).stem


def load_sroie():
    """darentang/sroie with an extra ``key`` column.

    Returns a DatasetDict: 'train' (626) and 'test' (347). Columns: id, key,
    words, bboxes (0-1000), ner_tags (ints, see LABELS), image_path.
    """
    ds = load_dataset(TAGGED_DATASET, revision="refs/convert/parquet")
    return ds.map(lambda row: {"key": receipt_key(row["image_path"])})


def load_official(split="test", with_images=True):
    """jsdnrs/ICDAR2019-SROIE split: key, image (PIL), image_size, entities
    (official answers), words, bboxes. Pass ``with_images=False`` to drop
    the image column (still downloads it, but the dataset is lighter to use)."""
    ds = load_dataset(OFFICIAL_DATASET, split=split)
    return ds if with_images else ds.remove_columns("image")


def load_images(keys, split="test"):
    """``{key: PIL.Image}`` for the given receipt keys (e.g. for figures)."""
    keys = set(keys)
    official = load_official(split)
    wanted = [i for i, k in enumerate(official["key"]) if k in keys]
    return {row["key"]: row["image"] for row in official.select(wanted)}
