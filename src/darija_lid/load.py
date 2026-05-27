"""Data loading utilities for the Darija LID benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from datasets import Dataset, DatasetDict, load_from_disk
import yaml


SOURCE_COUNT_KEY = "n_dataset"


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    classes: dict[str, int]
    final_fields: list[str]
    deduplicate_on: str
    sources: dict[str, str]


def repo_root_from_path(path: str | Path) -> Path:
    return Path(path).resolve().parents[1]


def normalize_dataset_name(dataset_name: str) -> str:
    if not isinstance(dataset_name, str):
        raise TypeError(
            f"Dataset name must be a string, got {type(dataset_name).__name__}"
        )

    if dataset_name.startswith("https://huggingface.co/datasets/"):
        parsed = urlparse(dataset_name)
        dataset_path = parsed.path.removeprefix("/datasets/").strip("/")
        if not dataset_path:
            raise ValueError(f"Invalid Hugging Face dataset URL: {dataset_name}")
        return dataset_path

    return dataset_name


def iter_sources(config: dict) -> list[tuple[str, str]]:
    sources = config.get("sources", {})

    if not isinstance(sources, dict):
        raise TypeError("'sources' must be a mapping in dataset.yaml")

    normalized_sources: list[tuple[str, str]] = []

    for source_name, source_config in sources.items():
        if source_name == SOURCE_COUNT_KEY:
            continue

        if isinstance(source_config, dict):
            dataset_name = normalize_dataset_name(source_config["dataset"])
            normalized_sources.append((source_name, dataset_name))
            continue

        if isinstance(source_config, str):
            normalized_sources.append(
                (source_name, normalize_dataset_name(source_config))
            )
            continue

        raise TypeError(
            f"Unsupported source config for '{source_name}': "
            f"expected dict or str, got {type(source_config).__name__}"
        )

    return normalized_sources


def load_project_config(config_path: str | Path) -> ProjectConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    classes = raw_config.get("classes", {})
    final_dataset = raw_config.get("final_dataset", {})
    final_fields = final_dataset.get(
        "fields",
        ["text", "dialect", "src", "has_latin_script", "label"],
    )
    deduplicate_on = final_dataset.get("deduplicate_on", "text")

    if not isinstance(classes, dict) or not classes:
        raise ValueError("'classes' must be defined in dataset.yaml")

    if not isinstance(final_fields, list) or not final_fields:
        raise ValueError("'final_dataset.fields' must be a non-empty list")

    sources = dict(iter_sources(raw_config))

    return ProjectConfig(
        seed=int(raw_config.get("seed", 42)),
        classes={str(name): int(value) for name, value in classes.items()},
        final_fields=[str(field) for field in final_fields],
        deduplicate_on=str(deduplicate_on),
        sources=sources,
    )


def raw_data_dir(repo_root: str | Path) -> Path:
    return Path(repo_root) / "data" / "raw"


def interim_data_dir(repo_root: str | Path) -> Path:
    return Path(repo_root) / "data" / "interim"


def prepared_sources_dir(repo_root: str | Path) -> Path:
    return interim_data_dir(repo_root) / "prepared_sources"


def processed_data_dir(repo_root: str | Path) -> Path:
    return Path(repo_root) / "data" / "processed"


def load_saved_dataset(path: str | Path) -> Dataset | DatasetDict:
    return load_from_disk(str(path))


def iter_splits(dataset_obj: Dataset | DatasetDict) -> list[tuple[str, Dataset]]:
    if isinstance(dataset_obj, DatasetDict):
        return list(dataset_obj.items())

    if isinstance(dataset_obj, Dataset):
        return [("train", dataset_obj)]

    raise TypeError(
        f"Unsupported dataset object type: {type(dataset_obj).__name__}"
    )
