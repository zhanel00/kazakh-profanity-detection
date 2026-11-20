"""Text cleaning helpers shared by data preparation, training and inference.

Nothing in this file is *learned* - these are fixed, rule-based functions, so they
can be applied to every split without leaking information from val/test.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# 1) Anonymisation (applied once, in prepare_data.py, before anything is saved)
# ---------------------------------------------------------------------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_PHONE = re.compile(r"(?<!\d)(?:\+?[78][\s\-()]*)?(?:\d[\s\-()]*){9,10}\d(?!\d)")
_HANDLE = re.compile(r"@(?!user\b)[A-Za-z0-9_.]{3,}")
# Surnames / patronymics: capitalised word whose stem ends in a typical Kazakh /
# Russian surname suffix, optionally followed by a Kazakh case ending.
_UP = "А-ЯӘІҢҒҮҰҚӨҺЁ"
_LO = "а-яәіңғүұқөһё"
_SURNAME = re.compile(
    rf"\b[{_UP}][{_LO}]{{2,15}}(?:ов|ев|ова|ева|овна|евна|ович|евич|ұлы|улы|қызы|кызы)"
    rf"(?:[{_LO}]{{0,5}})\b"
    rf"|\b[{_UP}]{{3,15}}(?:ОВ|ЕВ|ОВА|ЕВА|ҰЛЫ|ҚЫЗЫ)\b"
)
# capitalised words that look like surnames but are ordinary words
_NOT_NAMES = ("обнов", "овнов", "плов", "парков", "гурьев")


def anonymize(text: str) -> str:
    """Mask e-mails, URLs, phone numbers, user handles and surname-like tokens."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _HANDLE.sub("@user", text)

    def _mask(m):
        return m.group(0) if m.group(0).lower().startswith(_NOT_NAMES) else "[NAME]"

    text = _SURNAME.sub(_mask, text)
    text = _INTRO.sub(lambda m: m.group(1) + " [NAME]", text)
    text = _NEXT_TO_NAME.sub("[NAME]", text)
    return text


# "Есімім Бахтияр Нургиса" / "Атым Айзере ..." -> mask the self-introduced name
_INTRO = re.compile(rf"\b([Ее]сімім|[Ее]симим|[Аа]тым)\s+[{_UP}][{_LO}]+(?:\s+[{_UP}][{_LO}]+)?")
# first name written right next to a masked surname: "Тәттігүл [NAME]" -> "[NAME]"
_NEXT_TO_NAME = re.compile(
    rf"(?:\b[{_UP}][{_LO}{_UP}]+\s+)?\[NAME\](?:\s+[{_UP}][{_LO}{_UP}]+\b)?")


# ---------------------------------------------------------------------------
# 2) Model-input normalisation (used identically in training and inference)
# ---------------------------------------------------------------------------
_REPEAT = re.compile(r"(.)\1{3,}")  # 4+ identical chars in a row
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Light normalisation before the tokenizer.

    * NFKC unicode normalisation (merges look-alike code points)
    * lower-case (CAPS-LOCK swearing == normal swearing)
    * collapse 4+ repeated characters to 3 ("Қотааааақ" -> "қотааақ")
    * newlines/tabs -> single space

    Punctuation, emoji and masking symbols (*, @, #) are KEPT on purpose:
    'қ***қ' or 'з@еб@л' are masked profanity, and the tokenizer splits them into
    sub-word pieces that the model can learn.
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _REPEAT.sub(r"\1\1\1", text)
    return _SPACES.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# 3) Key used only for duplicate / near-duplicate detection
# ---------------------------------------------------------------------------
_NON_WORD = re.compile(r"[^\w*]+")
_REPEAT_ANY = re.compile(r"(.)\1+")


def dedup_key(text: str) -> str:
    """Aggressive normalisation: drop punctuation/emoji, collapse all repeats,
    and collapse repeated tokens ("мал мал мал" -> "мал")."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = _NON_WORD.sub(" ", t)
    t = _REPEAT_ANY.sub(r"\1", t)
    toks = t.split()
    out = [w for i, w in enumerate(toks) if i == 0 or w != toks[i - 1]]
    return " ".join(out)
