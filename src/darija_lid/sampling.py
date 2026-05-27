"""Sampling utilities for the Darija LID benchmark."""

from __future__ import annotations

from collections import defaultdict

from datasets import Dataset, concatenate_datasets

from .clean import has_latin_script, moroccan_label_for_text, normalize_text


def dataset_from_records(records: list[dict], columns: list[str]) -> Dataset:
    ordered_records = [{column: row[column] for column in columns} for row in records]
    return Dataset.from_list(ordered_records)


def merge_datasets(datasets: list[Dataset]) -> Dataset:
    if not datasets:
        raise ValueError("At least one prepared dataset is required")

    if len(datasets) == 1:
        return datasets[0]

    return concatenate_datasets(datasets)


def deduplicate_records(
    records: list[dict],
    classes: dict[str, int],
    text_field: str = "text",
) -> tuple[list[dict], dict]:
    grouped_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped_records[record[text_field]].append(record)

    deduplicated_records: list[dict] = []
    conflict_rows: list[dict] = []
    dropped_conflict_rows = 0

    arabic_label = classes["arabic_moroccan"]
    arabizi_label = classes["arabizi_moroccan"]
    other_label = classes["other"]

    for text in sorted(grouped_records):
        rows = grouped_records[text]
        src_values = sorted({row["src"] for row in rows})
        dialect_values = sorted({row["dialect"] for row in rows})
        label_values = sorted({int(row["label"]) for row in rows})
        contains_latin = any(bool(row["has_latin_script"]) for row in rows)
        contains_latin = contains_latin or has_latin_script(text)

        is_conflict = len(rows) > 1 and (
            len(label_values) > 1 or len(dialect_values) > 1
        )

        if is_conflict:
            conflict_rows.append(
                {
                    "text": text,
                    "labels": label_values,
                    "dialects": dialect_values,
                    "src": src_values,
                    "row_count": len(rows),
                }
            )
            dropped_conflict_rows += len(rows)
            continue

        if arabic_label in label_values or arabizi_label in label_values:
            resolved_dialect, contains_latin, resolved_label = moroccan_label_for_text(
                text, classes
            )
            if resolved_label not in label_values:
                resolved_label = (
                    arabic_label if arabic_label in label_values else arabizi_label
                )
                resolved_dialect = (
                    "arabic_moroccan"
                    if resolved_label == arabic_label
                    else "arabizi_moroccan"
                )
        else:
            resolved_label = other_label
            resolved_dialect = (
                dialect_values[0]
                if len(dialect_values) == 1
                else "|".join(dialect_values)
            )

        deduplicated_records.append(
            {
                "text": normalize_text(text),
                "dialect": resolved_dialect,
                "src": "|".join(src_values),
                "has_latin_script": contains_latin,
                "label": resolved_label,
            }
        )

    summary = {
        "input_rows": len(records),
        "output_rows": len(deduplicated_records),
        "duplicates_removed": len(records) - len(deduplicated_records),
        "conflict_count": len(conflict_rows),
        "dropped_conflict_rows": dropped_conflict_rows,
        "conflicts": conflict_rows,
    }
    return deduplicated_records, summary
