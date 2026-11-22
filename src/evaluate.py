"""Shared field-level evaluation for all methods (Regex, BERT, LayoutLM).

Every method saves its predictions as ``results/<method>_preds.json``, keyed
by the receipt key (image file name without extension, ``row["key"]``)::

    {"X51005230621": {"company": "...", "date": "...", "address": "...", "total": "..."}}

A missing field or an empty string means "no prediction" for that field.

Two gold standards (report both):

* ``get_gold(split)`` - **main score**. Gold strings rebuilt from the
  darentang word tags, i.e. what a perfect tagger could output from the OCR.
* ``get_official_gold(split, keys)`` - **second score**. The official ICDAR
  answers from jsdnrs/ICDAR2019-SROIE. Even a perfect tagger cannot reach 100 %
  here, because of OCR noise; ``evaluate(get_gold(test), official)`` gives that
  ceiling.

Scoring (per field): after normalisation, a prediction is correct only if it
exactly matches the gold string.

    precision = correct / receipts with a non-empty prediction
    recall    = correct / receipts with a non-empty gold value
    f1        = 2 * P * R / (P + R)

Usage (in a notebook, from the repo root)::

    from src.data import load_sroie
    from src.evaluate import get_gold, get_official_gold, bio_to_fields, evaluate, compare, save_preds

    test = load_sroie()["test"]
    gold = get_gold(test)                                   # main
    official = get_official_gold("test", keys=test["key"])  # second

    # Regex baseline: build {key: {field: string}} directly.
    # BERT / LayoutLM: turn word-level label predictions into fields:
    preds = {row["key"]: bio_to_fields(row["words"], word_labels) for row, word_labels in ...}

    save_preds(preds, "results/bert_preds.json")
    print(evaluate(preds, gold))
    print(evaluate(preds, official))

    # Issue #7 - one table comparing all methods (F1 per field):
    print(compare({"Regex": regex_preds, "BERT": bert_preds, "LayoutLM": lm_preds}, gold))

Command line::

    python -m src.evaluate results/regex_preds.json results/bert_preds.json
"""

import json
import re
import string
import sys

import pandas as pd

FIELDS = ["company", "date", "address", "total"]

_CURRENCY = re.compile(r"^(rm|myr|\$|usd|sgd)\s*")
_PUNCT_EDGES = string.punctuation.replace("$", "")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize(text, field=None):
    """Lowercase, collapse whitespace, strip punctuation at the ends.

    For ``total`` also drop currency prefixes and thousands separators,
    so "RM 1,234.50" == "1234.50".
    """
    if text is None:
        return ""
    text = " ".join(str(text).lower().split())
    text = text.strip(_PUNCT_EDGES + " ")
    if field == "total":
        text = _CURRENCY.sub("", text)
        text = text.replace(",", "").replace(" ", "")
    return text


# ---------------------------------------------------------------------------
# BIO labels -> field strings
# ---------------------------------------------------------------------------
def _field_of(label):
    """'B-COMPANY' -> ('B', 'company'); 'O' -> (None, None)."""
    if not label or label == "O" or "-" not in label:
        return None, None
    prefix, name = label.split("-", 1)
    name = name.lower()
    return (prefix.upper(), name) if name in FIELDS else (None, None)


def bio_to_fields(words, labels):
    """Convert word-level BIO labels into ``{field: string}``.

    ``labels`` are label strings (e.g. "B-TOTAL"), one per word. Use
    ``id2label`` first if you have integer ids.

    Words are grouped into spans: a ``B-`` tag, or an ``I-`` tag that does not
    continue the same field, starts a new span. A field can appear several times
    on one receipt (e.g. the total printed twice). Duplicate spans are therefore
    dropped, and the remaining spans are joined with a space in reading order.
    Gold and predictions go through this same function, so they are treated
    the same way.
    """
    if len(words) != len(labels):
        raise ValueError(f"{len(words)} words but {len(labels)} labels")

    spans = {f: [] for f in FIELDS}
    current_field, current_words = None, []

    def close():
        if current_field and current_words:
            spans[current_field].append(" ".join(current_words))

    for word, label in zip(words, labels):
        prefix, field = _field_of(label)
        if field is None:
            close()
            current_field, current_words = None, []
        elif prefix == "I" and field == current_field:
            current_words.append(word)
        else:
            close()
            current_field, current_words = field, [word]
    close()

    result = {}
    for field in FIELDS:
        unique = list(dict.fromkeys(spans[field]))  # dedupe, keep order
        result[field] = " ".join(unique)
    return result


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------
def _label_names(dataset_split):
    """Integer -> label-name list from a Hugging Face dataset, if available."""
    try:
        return dataset_split.features["ner_tags"].feature.names
    except (AttributeError, KeyError, TypeError):
        return None


def get_gold(dataset_split, label_names=None):
    """Build ``{receipt_key: {field: string}}`` from a labelled split.

    ``dataset_split`` is a Hugging Face split (or a list of dicts) with
    ``words`` and ``ner_tags``. Receipts are keyed by the ``key`` column
    (added by ``src.data.load_sroie``); without it, by ``id`` or row index.
    """
    names = label_names or _label_names(dataset_split)
    gold = {}
    for i, row in enumerate(dataset_split):
        tags = row["ner_tags"]
        if tags and not isinstance(tags[0], str):
            if names is None:
                raise ValueError("ner_tags are integers: pass label_names=[...]")
            tags = [names[t] for t in tags]
        gold[receipt_id(row, i)] = bio_to_fields(row["words"], tags)
    return gold


def receipt_id(row, index=None):
    """The key predictions are stored under: ``key``, else ``id``, else index."""
    for column in ("key", "id"):
        if column in row:
            return str(row[column])
    return str(index)


def get_official_gold(split="test", keys=None, official=None):
    """Official ICDAR answers: ``{receipt_key: {field: string}}``.

    Loads jsdnrs/ICDAR2019-SROIE unless ``official`` (an already loaded split)
    is given. Pass ``keys`` (e.g. ``test["key"]``) to keep only our receipts.
    """
    if official is None:
        from src.data import load_official
        official = load_official(split, with_images=False)
    keep = set(keys) if keys is not None else None
    gold = {}
    for key, entities in zip(official["key"], official["entities"]):
        if keep is None or key in keep:
            gold[key] = {f: entities.get(f) or "" for f in FIELDS}
    if keep is not None and len(gold) != len(keep):
        missing = sorted(keep - set(gold))[:5]
        raise ValueError(f"{len(keep) - len(gold)} keys not in official data, e.g. {missing}")
    return gold


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def evaluate(preds, gold, fields=FIELDS):
    """Per-field precision / recall / F1 (in %) plus a macro-average row.

    Only receipts in ``gold`` are scored. A receipt missing from ``preds``
    counts as "no prediction" for every field.
    """
    rows = []
    for field in fields:
        correct = n_pred = n_gold = 0
        for rid, gold_fields in gold.items():
            g = normalize(gold_fields.get(field), field)
            p = normalize(preds.get(rid, {}).get(field), field)
            n_gold += bool(g)
            n_pred += bool(p)
            correct += bool(p) and p == g
        precision = correct / n_pred if n_pred else 0.0
        recall = correct / n_gold if n_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "field": field,
            "precision": 100 * precision,
            "recall": 100 * recall,
            "f1": 100 * f1,
            "correct": correct,
            "predicted": n_pred,
            "gold": n_gold,
        })

    df = pd.DataFrame(rows).set_index("field")
    macro = df[["precision", "recall", "f1"]].mean()
    df.loc["macro_avg", ["precision", "recall", "f1"]] = macro
    df[["precision", "recall", "f1"]] = df[["precision", "recall", "f1"]].round(2)
    df[["correct", "predicted", "gold"]] = df[["correct", "predicted", "gold"]].astype("Int64")
    return df


def compare(method_preds, gold, fields=FIELDS):
    """F1 table: rows = fields + macro_avg, columns = methods.

    If both "BERT" and "LayoutLM" are present, adds a "layout_gain" column
    (LayoutLM F1 - BERT F1), which answers the research question.
    """
    table = pd.DataFrame({
        name: evaluate(preds, gold, fields)["f1"] for name, preds in method_preds.items()
    })
    if "BERT" in table and "LayoutLM" in table:
        table["layout_gain"] = (table["LayoutLM"] - table["BERT"]).round(2)
    return table


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_preds(preds, path):
    """Save predictions as JSON (keys converted to strings)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in preds.items()}, f, indent=2, ensure_ascii=False)


def load_preds(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(paths):
    from src.data import load_sroie

    test = load_sroie()["test"]
    golds = {
        "main gold (darentang tags)": get_gold(test),
        "official ICDAR answers": get_official_gold("test", keys=test["key"]),
    }
    for path in paths:
        preds = load_preds(path)
        for name, gold in golds.items():
            print(f"\n=== {path} vs {name} ===")
            print(evaluate(preds, gold).to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.evaluate results/<method>_preds.json [...]")
    main(sys.argv[1:])
