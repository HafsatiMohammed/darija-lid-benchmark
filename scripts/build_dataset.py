from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from darija_lid.load import (  # noqa: E402
    load_project_config,
    load_saved_dataset,
    prepared_sources_dir,
    processed_data_dir,
)
from darija_lid.sampling import (  # noqa: E402
    dataset_from_records,
    deduplicate_records,
    merge_datasets,
)


def write_csv(rows: list[dict], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_project_config(REPO_ROOT / "configs" / "dataset.yaml")
    prepared_dir = prepared_sources_dir(REPO_ROOT)
    output_dir = processed_data_dir(REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_datasets = []
    for source_key in config.sources:
        source_path = prepared_dir / source_key
        if not source_path.exists():
            raise FileNotFoundError(
                f"Prepared source is missing for '{source_key}': {source_path}"
            )
        prepared_datasets.append(load_saved_dataset(source_path))

    merged_dataset = merge_datasets(prepared_datasets)
    merged_records = [dict(row) for row in merged_dataset]
    final_records, dedup_summary = deduplicate_records(
        merged_records,
        classes=config.classes,
        text_field=config.deduplicate_on,
    )

    final_dataset = dataset_from_records(final_records, config.final_fields)
    dataset_output_path = output_dir / "final_dataset"
    csv_output_path = output_dir / "final_dataset.csv"
    summary_output_path = output_dir / "build_summary.json"
    conflicts_output_path = output_dir / "dropped_conflicts.json"

    if dataset_output_path.exists():
        shutil.rmtree(dataset_output_path)

    final_dataset.save_to_disk(str(dataset_output_path))
    write_csv(final_records, csv_output_path, config.final_fields)

    label_counts: dict[str, int] = {}
    for record in final_records:
        label_key = str(record["label"])
        label_counts[label_key] = label_counts.get(label_key, 0) + 1

    build_summary = {
        "row_count": len(final_records),
        "label_counts": label_counts,
        "deduplication": dedup_summary,
        "final_fields": config.final_fields,
    }
    with summary_output_path.open("w", encoding="utf-8") as file:
        json.dump(build_summary, file, ensure_ascii=False, indent=2)
    with conflicts_output_path.open("w", encoding="utf-8") as file:
        json.dump(
            dedup_summary.get("conflicts", []),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved final dataset to {dataset_output_path}")
    print(f"Saved final dataset CSV to {csv_output_path}")
    print(f"Saved build summary to {summary_output_path}")
    print(f"Saved dropped conflicts to {conflicts_output_path}")


if __name__ == "__main__":
    main()
