"""Method 1: rule-based (regex) field extraction from receipt OCR words.

SROIE already provides OCR words and their boxes, so the "OCR" step is the
dataset's own OCR output. We group words into text lines, then apply
hand-written rules for each field.
"""

import re

# --- Patterns ---------------------------------------------------------------
# Hand-written patterns for each field. They encode what a person would look for on a Malaysian
# receipt; this baseline shows how far rules get without any learning.

# Month names and their abbreviations ("Dec", "December", "Sept").
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
# Date formats seen on the receipts, tried in this order (numeric formats are by far the most common).
DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b"),      # 25/12/2018, 12-01-19
    re.compile(r"\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b"),        # 2018-12-25
    re.compile(rf"\b\d{{1,2}}[\s\-]?{_MONTHS}[\s\-,]*\d{{2,4}}\b", re.I),  # 25 Dec 2018
    re.compile(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+\d{{2,4}}\b", re.I),        # Dec 25, 2018
]
# A money amount with exactly two decimals, optionally with thousands separators ("1,234.50").
# "39. 80": OCR sometimes inserts a space after the decimal point.
AMOUNT = re.compile(r"\d{1,3}(?:,\d{3})+\.\s?\d{2}\b|\d+\.\s?\d{2}\b")
# Words that mark the line carrying the amount to pay.
TOTAL_LINE = re.compile(r"total|amount\s*due|nett|grand|amount\s*payable", re.I)
# Total lines that are clearly the final amount (after tax / rounding), preferred over a plain "TOTAL".
STRONG_TOTAL = re.compile(r"incl(?!udes)|rounded|grand|nett|net\s*total|amount\s*due|payable", re.I)
# Lines that mention "total" but are not the amount to pay (subtotal, tax, quantity, change, ...).
NOT_TOTAL = re.compile(
    r"sub\s*-?\s*total|total\s*(qty|quantity|item|gst|tax|saving|discount|disc)|"
    r"excl|change|tender|qty|rounding\s*adj|before", re.I)
# "Rounding adjustment" lines: Malaysian receipts round the total to 5 sen.
ROUNDING = re.compile(r"round", re.I)
# An amount with an optional minus sign / "RM", used to read a rounding adjustment such as "-0.02".
SIGNED_AMOUNT = re.compile(r"(-)?\s*(?:rm)?\s*(\d+\.\s?\d{2})\b", re.I)
# Words that usually appear in a business name (company suffixes, shop types, Malay terms).
COMPANY_HINT = re.compile(
    r"\b(sdn\.?\s*bhd|bhd|s/b|enterprise|trading|restaurant|restoran|store|mart|"
    r"hardware|plt|co\b|company|marketing|shop|supermarket|bakery|cafe|"
    r"kedai|perniagaan|station|services|industries|hotel|pharmacy|book|"
    r"stationery|electrical|motor|machinery|gift|deco|restaurants)", re.I)
# Header lines near the top that are not the company name.
SKIP_LINE = re.compile(r"copy|tax\s*invoice|invoice|receipt|welcome|cash\s*bill|"
                       r"\bgst\b|reg\.?\s*no|roc|co\.?\s*no|tel|fax", re.I)
# Malaysian postcodes have 5 digits; the postcode line usually ends the address.
POSTCODE = re.compile(r"\b\d{5}\b")
# Lines that come after the address block (contact details, invoice data), so the address stops there.
ADDRESS_STOP = re.compile(r"tel|fax|gst|invoice|receipt|e-?mail|www|reg|roc|"
                          r"date|cashier|bill|counter|h/p|phone", re.I)


# --- Lines ------------------------------------------------------------------
def group_lines(words, bboxes):
    """Group consecutive words into text lines using their vertical position.

    A new line starts when a word's top edge moves by more than half the
    previous word's height, or when the next word jumps back to the left.
    """
    # The rules work on lines ("Total : 9.00"), but the dataset gives single words, so rebuild the lines.
    lines, current = [], []
    prev = None
    for word, (x0, y0, x1, y1) in zip(words, bboxes):
        if prev is not None:
            px0, py0, px1, py1 = prev
            height = max(py1 - py0, 1)                       # avoid a zero height for degenerate boxes
            if abs(y0 - py0) > height / 2 or x0 < px0:       # moved down a line, or wrapped back to the left
                lines.append(" ".join(current))
                current = []
        current.append(word)
        prev = (x0, y0, x1, y1)
    if current:
        lines.append(" ".join(current))
    return lines


def _to_float(amount):
    """'1,234.50' or '39. 80' -> float, for comparing and adding amounts."""
    return float(amount.replace(",", "").replace(" ", ""))


# --- Field rules ------------------------------------------------------------
def extract_date(lines):
    """First date found, scanning lines top to bottom (the receipt date is printed near the top)."""
    for line in lines:
        for pattern in DATE_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(0)
    return ""


def extract_total(lines):
    """Amount on a 'TOTAL' line (not subtotal/GST/qty...), or on the line below.

    1. A TOTAL line right after a "Rounding" line is the final rounded total.
    2. Else lines like "Total Incl. GST" / "Total Rounded" / "Grand Total" win.
    3. Else the first plain TOTAL line, plus any rounding adjustment below it.
    Tax-summary rows with two different amounts are skipped.
    Fallback: the largest amount on the receipt.
    """
    candidates = []  # (line index, amount, is_strong)
    for i, line in enumerate(lines):
        if not TOTAL_LINE.search(line) or NOT_TOTAL.search(line):
            continue
        amounts = AMOUNT.findall(line)
        if not amounts and i + 1 < len(lines):
            amounts = AMOUNT.findall(lines[i + 1])           # label and amount sometimes split over two lines
        if len(set(amounts)) != 1:
            continue                                         # none, or a tax-summary row with several amounts
        candidates.append((i, amounts[0], bool(STRONG_TOTAL.search(line))))

    roundings = [(i, line) for i, line in enumerate(lines)
                 if ROUNDING.search(line) and not TOTAL_LINE.search(line)]

    # Rule 1: a total printed after the rounding line is already the rounded amount to pay.
    if roundings:
        after = [c for c in candidates if c[0] > roundings[0][0]]
        if after:
            return after[0][1]
    # Rule 2: the last explicit final-total line ("Total Incl. GST", "Grand Total", ...).
    strong = [c for c in candidates if c[2]]
    if strong:
        return strong[-1][1]
    # Rule 3: the first plain total, corrected by a rounding adjustment printed below it.
    if candidates:
        idx, amount = candidates[0][:2]
        for r_idx, r_line in roundings:
            if r_idx > idx:
                match = SIGNED_AMOUNT.search(r_line)
                if match:
                    adj = _to_float(match.group(2)) * (-1 if match.group(1) else 1)
                    return f"{_to_float(amount) + adj:.2f}"
        return amount
    # Fallback: the total is usually the largest amount on the receipt.
    all_amounts = [a for line in lines for a in AMOUNT.findall(line)]
    if all_amounts:
        return max(all_amounts, key=_to_float)
    return ""


def _company_index(lines, search_lines=8):
    """Index of the company line: first line (near the top) with a company
    keyword, else the first line that looks like a name."""
    # The company name is printed at the top, so only the first few lines are searched.
    top = lines[:search_lines]
    for i, line in enumerate(top):
        if COMPANY_HINT.search(line) and not SKIP_LINE.search(line):
            return i
    # No keyword: take the first mostly-alphabetic line, i.e. a name rather than a number or code.
    for i, line in enumerate(top):
        letters = sum(c.isalpha() for c in line)
        if letters >= 4 and letters / max(len(line), 1) > 0.6 and not SKIP_LINE.search(line):
            return i
    return 0 if lines else None


def extract_company(lines):
    """Company name: the company line, joined with its continuation and cleaned of codes at the ends."""
    i = _company_index(lines)
    if i is None:
        return ""
    line = lines[i]
    # "SDN BHD" alone on its own line usually ends a name started on the line above.
    if i > 0 and re.fullmatch(r"\(?[\w\s.]*(sdn\.?\s*bhd|bhd)\.?\)?", line, re.I) and len(line.split()) <= 4:
        line = lines[i - 1] + " " + line
    # A name ending in "&" continues on the next line.
    if line.rstrip().endswith("&") and i + 1 < len(lines):
        line = line + " " + lines[i + 1]
    # Drop store codes / registration numbers at either end, e.g.
    # "A03018 UNIHAKKA ... BHD", "AEON CO. (M) BHD (126926-H)".
    words = line.split()
    while len(words) > 1 and (re.search(r"\d", words[0]) or re.fullmatch(r"a[o0]|/", words[0], re.I)) \
            and not COMPANY_HINT.search(words[0]):
        words.pop(0)
    while len(words) > 1 and re.search(r"\d", words[-1]):
        words.pop()
    return " ".join(words)


# Company registration numbers like "(JM0517726)", "( 801580-T )", "JM0325956-V".
REG_NO = re.compile(r"^\(?\s*[A-Z]{0,3}\d{5,}(-?\s?[A-Z])?\s*\)?[,.]?\s*", re.I)
# A short line of letters only, e.g. the state name "JOHOR." printed after the postcode line.
STATE_LINE = re.compile(r"^[A-Za-z .,()]+$")


def extract_address(lines, max_lines=5):
    """Lines after the company until (and including) the line with a 5-digit
    postcode, plus a short state-name line right after it ("JOHOR.").
    Registration numbers before the address and phone/GST/invoice lines are skipped."""
    # The address directly follows the company name, so start there and read at most max_lines lines.
    i = _company_index(lines)
    if i is None:
        return ""
    if lines[i].rstrip().endswith("&"):  # company name continued on the next line
        i += 1
    picked = []
    rest = lines[i + 1:]
    for j, line in enumerate(rest[:max_lines]):
        if not picked:
            # The registration number often sits between the name and the address: strip it.
            line = REG_NO.sub("", line).strip()
            if not line:
                continue
        if ADDRESS_STOP.search(line) and not POSTCODE.search(line):
            if picked:
                break                                        # contact/invoice details: the address has ended
            continue                                         # such a line before the address: skip it
        picked.append(line)
        if POSTCODE.search(line):
            # The postcode line ends the address; keep a following state-name line if there is one.
            nxt = rest[j + 1] if j + 1 < len(rest) else ""
            if nxt and len(nxt.split()) <= 2 and STATE_LINE.match(nxt) and not ADDRESS_STOP.search(nxt):
                picked.append(nxt)
            break
    return " ".join(picked)


def extract_fields(words, bboxes):
    """All four fields for one receipt, from its OCR words and boxes."""
    lines = group_lines(words, bboxes)
    return {
        "company": extract_company(lines),
        "date": extract_date(lines),
        "address": extract_address(lines),
        "total": extract_total(lines),
    }


def predict(dataset_split):
    """``{receipt_key: {field: string}}`` for a whole split (keyed like ``get_gold``)."""
    # Same keys as get_gold, so evaluate() can match each prediction to its gold receipt.
    from src.evaluate import receipt_id
    return {receipt_id(row, i): extract_fields(row["words"], row["bboxes"])
            for i, row in enumerate(dataset_split)}
