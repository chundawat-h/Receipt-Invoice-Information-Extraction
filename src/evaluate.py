"""Shared field-level evaluation for all methods (Regex, BERT, LayoutLM).

Implemented in Issue #2. Every method must save predictions as
results/<method>_preds.json in this format:

    {"receipt_id": {"company": "...", "date": "...", "address": "...", "total": "..."}}
"""

FIELDS = ["company", "date", "address", "total"]


def get_gold(dataset_split):
    """Build gold field strings per receipt from BIO-labelled words. (Issue #2)"""
    raise NotImplementedError


def bio_to_fields(words, labels):
    """Convert word-level BIO labels into {field: string}. (Issue #2)"""
    raise NotImplementedError


def evaluate(preds, gold):
    """Return a DataFrame of per-field precision / recall / F1. (Issue #2)"""
    raise NotImplementedError
