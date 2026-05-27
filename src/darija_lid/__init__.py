"""Darija LID benchmark package."""

from .clean import has_latin_script, moroccan_label_for_text, normalize_text
from .load import (
    ProjectConfig,
    iter_splits,
    load_project_config,
    load_saved_dataset,
    normalize_dataset_name,
    prepared_sources_dir,
    processed_data_dir,
    raw_data_dir,
    repo_root_from_path,
)
from .sampling import dataset_from_records, deduplicate_records, merge_datasets
from .validation import ValidationReport, validate_final_dataset

__all__ = [
    "ProjectConfig",
    "ValidationReport",
    "dataset_from_records",
    "deduplicate_records",
    "has_latin_script",
    "iter_splits",
    "load_project_config",
    "load_saved_dataset",
    "merge_datasets",
    "moroccan_label_for_text",
    "normalize_dataset_name",
    "normalize_text",
    "prepared_sources_dir",
    "processed_data_dir",
    "raw_data_dir",
    "repo_root_from_path",
    "validate_final_dataset",
]
