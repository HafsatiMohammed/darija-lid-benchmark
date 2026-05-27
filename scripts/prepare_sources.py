from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from darija_lid.clean import (  # noqa: E402
    has_latin_script,
    moroccan_label_for_text,
    normalize_other_dialect,
    normalize_text,
)
from darija_lid.load import (  # noqa: E402
    iter_splits,
    load_project_config,
    load_saved_dataset,
    prepared_sources_dir,
    raw_data_dir,
)
from darija_lid.sampling import dataset_from_records  # noqa: E402


def build_row(text: str, dialect: str, src: str, label: int) -> dict | None:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return None

    return {
        "text": normalized_text,
        "dialect": dialect,
        "src": src,
        "has_latin_script": has_latin_script(normalized_text),
        "label": label,
    }


def build_moroccan_row(
    text: str, src: str, classes: dict[str, int]
) -> dict | None:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return None

    dialect, contains_latin, label = moroccan_label_for_text(
        normalized_text, classes
    )
    return {
        "text": normalized_text,
        "dialect": dialect,
        "src": src,
        "has_latin_script": contains_latin,
        "label": label,
    }


def build_other_row(
    text: str,
    dialect: str,
    src: str,
    classes: dict[str, int],
) -> dict | None:
    row = build_row(
        text=text,
        dialect=normalize_other_dialect(dialect),
        src=src,
        label=classes["other"],
    )
    return row


def prepare_dataset0(
    source_path: Path,
    src_label: str,
    classes: dict[str, int],
) -> list[dict]:
    dataset_obj = load_saved_dataset(source_path)
    records: list[dict] = []

    for _, split_data in iter_splits(dataset_obj):
        selected = split_data.select_columns(["content", "is_darija"])
        for row in selected:
            if str(row["is_darija"]).strip().lower() == "yes":
                record = build_moroccan_row(row["content"], src_label, classes)
            else:
                record = build_other_row(row["content"], "other", src_label, classes)

            if record:
                records.append(record)

    return records


def prepare_dataset1(
    source_path: Path,
    src_label: str,
    classes: dict[str, int],
) -> list[dict]:
    dataset_obj = load_saved_dataset(source_path)
    records: list[dict] = []

    for _, split_data in iter_splits(dataset_obj):
        selected = split_data.select_columns(
            ["darija_Arab_new", "darija_Latn", "english"]
        )
        for row in selected:
            arabic_record = build_moroccan_row(
                row["darija_Arab_new"], src_label, classes
            )
            latin_record = build_moroccan_row(
                row["darija_Latn"], src_label, classes
            )
            english_record = build_other_row(
                row["english"], "english", src_label, classes
            )

            for record in (arabic_record, latin_record, english_record):
                if record:
                    records.append(record)

    return records


def prepare_dataset2(
    source_path: Path,
    src_label: str,
    classes: dict[str, int],
) -> list[dict]:
    dataset_obj = load_saved_dataset(source_path)
    records: list[dict] = []

    for _, split_data in iter_splits(dataset_obj):
        selected = split_data.select_columns(["dialect", "text", "label"])
        for row in selected:
            if str(row["label"]).strip().lower() == "ary":
                record = build_moroccan_row(row["text"], src_label, classes)
            else:
                record = build_other_row(
                    row["text"], row["dialect"], src_label, classes
                )

            if record:
                records.append(record)

    return records


def prepare_dataset3(
    source_path: Path,
    src_label: str,
    classes: dict[str, int],
) -> list[dict]:
    records: list[dict] = []

    for config_dir in sorted(source_path.iterdir()):
        if not config_dir.is_dir():
            continue

        dataset_obj = load_saved_dataset(config_dir)
        for _, split_data in iter_splits(dataset_obj):
            selected = split_data.select_columns(
                [
                    "country",
                    "dialect",
                    "english_conversation",
                    "dialectal_conversation",
                ]
            )
            for row in selected:
                is_moroccan = str(row["country"]).strip().upper() == "MA"

                for turn in row["dialectal_conversation"]:
                    text = turn.get("text")
                    if is_moroccan:
                        record = build_moroccan_row(text, src_label, classes)
                    else:
                        record = build_other_row(
                            text, row["dialect"], src_label, classes
                        )

                    if record:
                        records.append(record)

                for turn in row["english_conversation"]:
                    record = build_other_row(
                        turn.get("text"), "english", src_label, classes
                    )
                    if record:
                        records.append(record)

    return records


def prepare_dataset4(
    source_path: Path,
    src_label: str,
    classes: dict[str, int],
) -> list[dict]:
    dataset_obj = load_saved_dataset(source_path)
    records: list[dict] = []

    for _, split_data in iter_splits(dataset_obj):
        selected = split_data.select_columns(["dialect", "text"])
        for row in selected:
            record = build_other_row(
                row["text"], row["dialect"], src_label, classes
            )
            if record:
                records.append(record)

    return records


def summarize_records(source_key: str, src_label: str, records: list[dict]) -> dict:
    label_counts = {
        str(label): sum(1 for record in records if int(record["label"]) == label)
        for label in sorted({int(record["label"]) for record in records})
    }
    return {
        "source_key": source_key,
        "src": src_label,
        "rows": len(records),
        "label_counts": label_counts,
    }


def write_summary(summary_rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source_key", "src", "rows", "label_counts"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main() -> None:
    config = load_project_config(REPO_ROOT / "configs" / "dataset.yaml")
    raw_dir = raw_data_dir(REPO_ROOT)
    output_root = prepared_sources_dir(REPO_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    prepare_functions = {
        "dataset0": prepare_dataset0,
        "dataset1": prepare_dataset1,
        "dataset2": prepare_dataset2,
        "dataset3": prepare_dataset3,
        "dataset4": prepare_dataset4,
    }

    summary_rows: list[dict] = []

    for source_key, src_label in config.sources.items():
        prepare_fn = prepare_functions.get(source_key)
        if prepare_fn is None:
            raise KeyError(f"No prepare function is defined for '{source_key}'")

        source_path = raw_dir / source_key
        if not source_path.exists():
            raise FileNotFoundError(
                f"Raw source path does not exist for '{source_key}': {source_path}"
            )

        print(f"Preparing {src_label} from {source_path}")
        records = prepare_fn(source_path, src_label, config.classes)
        output_path = output_root / source_key

        if output_path.exists():
            shutil.rmtree(output_path)

        dataset = dataset_from_records(records, config.final_fields)
        dataset.save_to_disk(str(output_path))
        summary_rows.append(summarize_records(source_key, src_label, records))
        print(f"Saved {len(records)} prepared rows to {output_path}")

    summary_path = output_root / "prepared_sources_summary.csv"
    write_summary(summary_rows, summary_path)
    print(f"Saved preparation summary to {summary_path}")


if __name__ == "__main__":
    main()
