"""Validation utilities for the Darija LID benchmark."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from datasets import Dataset

from .clean import has_latin_script, normalize_text


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["is_valid"] = self.is_valid
        return data


def validate_final_dataset(
    dataset: Dataset,
    classes: dict[str, int],
    required_fields: list[str],
) -> ValidationReport:
    report = ValidationReport()

    dataset_columns = list(dataset.column_names)
    missing_columns = [field for field in required_fields if field not in dataset_columns]
    extra_columns = [field for field in dataset_columns if field not in required_fields]

    if missing_columns:
        report.errors.append(f"Missing required columns: {missing_columns}")

    if extra_columns:
        report.warnings.append(f"Extra columns found: {extra_columns}")

    label_counts: Counter[int] = Counter()
    duplicate_counts: Counter[str] = Counter()
    allowed_labels = set(classes.values())

    for index, row in enumerate(dataset):
        text = normalize_text(row.get("text"))
        dialect = str(row.get("dialect", "")).strip()
        src = str(row.get("src", "")).strip()
        has_latin = bool(row.get("has_latin_script"))
        label = row.get("label")

        if not text:
            report.errors.append(f"Row {index} has empty text")
            continue

        if not dialect:
            report.errors.append(f"Row {index} has empty dialect")

        if not src:
            report.errors.append(f"Row {index} has empty src")

        if label not in allowed_labels:
            report.errors.append(f"Row {index} has invalid label: {label}")
            continue

        expected_latin_flag = has_latin_script(text)
        if has_latin != expected_latin_flag:
            report.errors.append(
                f"Row {index} has inconsistent has_latin_script for text: {text[:80]}"
            )

        if label == classes["arabizi_moroccan"] and not has_latin:
            report.errors.append(
                f"Row {index} is arabizi_moroccan but has_latin_script is false"
            )

        if label == classes["arabic_moroccan"] and dialect != "arabic_moroccan":
            report.errors.append(
                f"Row {index} has label arabic_moroccan but dialect='{dialect}'"
            )

        if label == classes["arabizi_moroccan"] and dialect != "arabizi_moroccan":
            report.errors.append(
                f"Row {index} has label arabizi_moroccan but dialect='{dialect}'"
            )

        if label == classes["other"] and dialect in {
            "arabic_moroccan",
            "arabizi_moroccan",
        }:
            report.errors.append(
                f"Row {index} has label other but dialect='{dialect}'"
            )

        label_counts[int(label)] += 1
        duplicate_counts[text] += 1

    duplicate_texts = sum(1 for count in duplicate_counts.values() if count > 1)
    if duplicate_texts:
        report.errors.append(
            f"Found {duplicate_texts} duplicate texts after deduplication"
        )

    report.stats = {
        "row_count": len(dataset),
        "label_counts": dict(label_counts),
        "duplicate_texts": duplicate_texts,
    }
    return report
