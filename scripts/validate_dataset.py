from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from darija_lid.load import (  # noqa: E402
    load_project_config,
    load_saved_dataset,
    processed_data_dir,
)
from darija_lid.validation import validate_final_dataset  # noqa: E402


def main() -> None:
    config = load_project_config(REPO_ROOT / "configs" / "dataset.yaml")
    processed_dir = processed_data_dir(REPO_ROOT)
    dataset_path = processed_dir / "final_dataset"
    report_path = processed_dir / "validation_report.json"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Final dataset is missing. Run scripts/build_dataset.py first: {dataset_path}"
        )

    dataset = load_saved_dataset(dataset_path)
    report = validate_final_dataset(dataset, config.classes, config.final_fields)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, ensure_ascii=False, indent=2)

    print(f"Saved validation report to {report_path}")
    print(f"Rows: {report.stats.get('row_count', 0)}")
    print(f"Label counts: {report.stats.get('label_counts', {})}")

    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"- {warning}")

    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()
