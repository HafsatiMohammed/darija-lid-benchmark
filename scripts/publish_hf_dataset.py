from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huggingface_hub import HfApi  # noqa: E402

from darija_lid.load import load_project_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the final dataset to the Hugging Face Hub with a dataset card "
            "and viewer-compatible CSV layout."
        )
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target dataset repository, for example username/darija-lid-final",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Target branch on the Hub. Default: main",
    )
    parser.add_argument(
        "--commit-message",
        default="Publish final Darija LID dataset",
        help="Commit message for the Hub upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repository as private if it does not exist",
    )
    parser.add_argument(
        "--csv-path",
        default=str(REPO_ROOT / "data" / "processed" / "final_dataset.csv"),
        help="Path to the final CSV file to publish",
    )
    parser.add_argument(
        "--split-name",
        default="test",
        help="Dataset split name to publish in the data viewer. Default: test",
    )
    parser.add_argument(
        "--build-summary-path",
        default=str(REPO_ROOT / "data" / "processed" / "build_summary.json"),
        help="Path to build_summary.json",
    )
    parser.add_argument(
        "--validation-report-path",
        default=str(REPO_ROOT / "data" / "processed" / "validation_report.json"),
        help="Path to validation_report.json",
    )
    parser.add_argument(
        "--dropped-conflicts-path",
        default=str(REPO_ROOT / "data" / "processed" / "dropped_conflicts.json"),
        help="Path to dropped_conflicts.json",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(REPO_ROOT / "data" / "processed" / "plots"),
        help="Optional directory containing generated plot images to upload",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, huggingface_hub default auth is used.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def build_readme(
    repo_id: str,
    config,
    build_summary: dict,
    validation_report: dict,
    dropped_conflicts: list[dict],
    plot_paths: list[Path],
    split_name: str,
) -> str:
    label_name_by_id = {value: key for key, value in config.classes.items()}
    label_lines = []
    for label_id in sorted(label_name_by_id):
        label_name = label_name_by_id[label_id]
        label_lines.append(f"- `{label_id}`: `{label_name}`")

    label_count_lines = []
    for label_id_str, count in sorted(
        build_summary.get("label_counts", {}).items(),
        key=lambda item: int(item[0]),
    ):
        label_name = label_name_by_id.get(int(label_id_str), "unknown")
        label_count_lines.append(f"- `{label_name}` (`{label_id_str}`): {count:,}")

    sources_lines = [
        f"- `{source_name}`" for _, source_name in sorted(config.sources.items())
    ]

    dedup = build_summary.get("deduplication", {})
    middle_conflict_examples: list[dict] = []
    if dropped_conflicts:
        example_count = min(5, len(dropped_conflicts))
        start_index = max(0, (len(dropped_conflicts) - example_count) // 2)
        middle_conflict_examples = dropped_conflicts[
            start_index : start_index + example_count
        ]
    conflict_example_lines = []
    for conflict in middle_conflict_examples:
        conflict_example_lines.append(
            "- "
            f"`{conflict.get('text', '')}` "
            f"(labels={conflict.get('labels', [])}, "
            f"dialects={conflict.get('dialects', [])}, "
            f"rows={conflict.get('row_count', 0)})"
        )
    conflict_example_block = "\n".join(conflict_example_lines) or "- none"
    plot_section = ""
    if plot_paths:
        plot_lines = []
        for plot_path in plot_paths:
            rel_path = f"./plots/{plot_path.name}"
            plot_title = plot_path.stem.replace("_", " ").title()
            plot_lines.append(f"### {plot_title}\n\n![{plot_title}]({rel_path})")
        plot_section = "\n\n## Visual Summary\n\n" + "\n\n".join(plot_lines)

    return f"""---
pretty_name: Darija LID Final Dataset
license: mit
task_categories:
- text-classification
language:
- ar
- fr
- en
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: {split_name}
    path: {split_name}.csv
---

# {repo_id}

This repository contains the final merged Darija language identification dataset prepared from multiple public sources.

## What Was Done

- Downloaded the configured source datasets.
- Flattened each source into one shared schema: `text`, `dialect`, `src`, `has_latin_script`, `label`.
- Mapped labels to three classes:
{chr(10).join(label_lines)}
- Added English translation rows when they were explicitly available in source datasets.
- Kept Moroccan Darija in two buckets:
  - Arabic-script Moroccan
  - Arabizi / Latin-script Moroccan
- Removed repeated normalized lines so the same text appears only once in the final dataset.
- Preserved provenance in the `src` column. If a row came from multiple sources, `src` stores them joined by `|`.

## Dataset Schema

- `text`: normalized text
- `dialect`: normalized dialect name
- `src`: source dataset identifier(s)
- `has_latin_script`: whether the row contains at least one Latin character
- `label`: numeric class ID

## Label Counts

{chr(10).join(label_count_lines)}

## Build Summary

- Final row count: {build_summary.get("row_count", 0):,}
- Input rows before deduplication: {dedup.get("input_rows", 0):,}
- Output rows after deduplication: {dedup.get("output_rows", 0):,}
- Duplicate rows removed: {dedup.get("duplicates_removed", 0):,}
- Duplicate groups with label or dialect conflicts: {dedup.get("conflict_count", 0):,}
- Raw rows excluded because they belonged to conflict groups: {dedup.get("dropped_conflict_rows", 0):,}

## Example Conflict Cases

Sampled from the middle of the conflict list:

{conflict_example_block}

## Source Datasets

{chr(10).join(sources_lines)}

## Notes

- This is a text classification dataset for Darija language identification.
- The `other` class includes non-Moroccan dialects, English translation rows, and the remaining non-target content.
- The Hugging Face data viewer is enabled through the `{split_name}.csv` file declared above.
{plot_section}
"""


def prepare_upload_folder(
    upload_dir: Path,
    csv_path: Path,
    build_summary_path: Path,
    validation_report_path: Path,
    dropped_conflicts_path: Path,
    readme_text: str,
    plot_paths: list[Path],
    split_name: str,
) -> None:
    shutil.copy2(csv_path, upload_dir / f"{split_name}.csv")
    artifacts_dir = upload_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build_summary_path, artifacts_dir / "build_summary.json")
    shutil.copy2(validation_report_path, artifacts_dir / "validation_report.json")
    shutil.copy2(dropped_conflicts_path, artifacts_dir / "dropped_conflicts.json")
    if plot_paths:
        plots_dir = upload_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for plot_path in plot_paths:
            shutil.copy2(plot_path, plots_dir / plot_path.name)
    (upload_dir / "README.md").write_text(readme_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_project_config(REPO_ROOT / "configs" / "dataset.yaml")

    csv_path = Path(args.csv_path)
    build_summary_path = Path(args.build_summary_path)
    validation_report_path = Path(args.validation_report_path)
    dropped_conflicts_path = Path(args.dropped_conflicts_path)
    plots_dir = Path(args.plots_dir)

    ensure_exists(csv_path)
    ensure_exists(build_summary_path)
    ensure_exists(validation_report_path)
    ensure_exists(dropped_conflicts_path)

    plot_paths: list[Path] = []
    if plots_dir.exists():
        plot_paths = sorted(
            path
            for path in plots_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
        )

    build_summary = load_json(build_summary_path)
    validation_report = load_json(validation_report_path)
    dropped_conflicts = load_json(dropped_conflicts_path)
    readme_text = build_readme(
        repo_id=args.repo_id,
        config=config,
        build_summary=build_summary,
        validation_report=validation_report,
        dropped_conflicts=dropped_conflicts,
        plot_paths=plot_paths,
        split_name=args.split_name,
    )

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(prefix="darija_lid_publish_") as tmp_dir:
        upload_dir = Path(tmp_dir)
        prepare_upload_folder(
            upload_dir=upload_dir,
            csv_path=csv_path,
            build_summary_path=build_summary_path,
            validation_report_path=validation_report_path,
            dropped_conflicts_path=dropped_conflicts_path,
            readme_text=readme_text,
            plot_paths=plot_paths,
            split_name=args.split_name,
        )
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(upload_dir),
            revision=args.branch,
            commit_message=args.commit_message,
        )

    print(f"Published dataset to https://huggingface.co/datasets/{args.repo_id}")
    if plot_paths:
        print("Uploaded plot images:")
        for plot_path in plot_paths:
            print(f"- plots/{plot_path.name}")
    else:
        print(f"No plot images found in {plots_dir}")


if __name__ == "__main__":
    main()
