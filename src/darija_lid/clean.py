"""Text cleaning utilities for the Darija LID benchmark."""

from __future__ import annotations

import re


LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if not isinstance(text, str):
        return ""

    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    return normalized.split(" ")


def has_latin_script(text: str | None) -> bool:
    return bool(LATIN_RE.search(normalize_text(text)))


def has_arabic_script(text: str | None) -> bool:
    return bool(ARABIC_RE.search(normalize_text(text)))


def normalize_other_dialect(dialect: str | None) -> str:
    normalized = normalize_text(dialect).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized or "other"


def moroccan_label_for_text(text: str, classes: dict[str, int]) -> tuple[str, bool, int]:
    normalized_text = normalize_text(text)
    tokens = tokenize(normalized_text)
    arabic_token_count = sum(1 for token in tokens if has_arabic_script(token))
    latin_token_count = sum(1 for token in tokens if has_latin_script(token))
    contains_latin = latin_token_count > 0

    if latin_token_count > 0 and arabic_token_count == 0:
        return (
            "arabizi_moroccan",
            contains_latin,
            classes["arabizi_moroccan"],
        )

    if latin_token_count > arabic_token_count:
        return (
            "arabizi_moroccan",
            contains_latin,
            classes["arabizi_moroccan"],
        )

    return (
        "arabic_moroccan",
        contains_latin,
        classes["arabic_moroccan"],
    )
