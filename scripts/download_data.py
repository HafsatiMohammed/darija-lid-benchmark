from pathlib import Path
from urllib.parse import urlparse

from datasets import get_dataset_config_names, load_dataset
import yaml


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


def iter_sources(config: dict) -> list[tuple[str, str, list[str] | None]]:
    sources = config.get("sources", {})

    if not isinstance(sources, dict):
        raise TypeError("'sources' must be a mapping in dataset.yaml")

    normalized_sources: list[tuple[str, str, list[str] | None]] = []

    for source_name, source_config in sources.items():
        if source_name == "n_dataset":
            continue

        if isinstance(source_config, dict):
            dataset_name = normalize_dataset_name(source_config["dataset"])
            splits = [split for split in source_config.get("splits", []) if split]
            normalized_sources.append((source_name, dataset_name, splits or None))
            continue

        if isinstance(source_config, str):
            normalized_sources.append(
                (source_name, normalize_dataset_name(source_config), None)
            )
            continue

        raise TypeError(
            f"Unsupported source config for '{source_name}': "
            f"expected dict or str, got {type(source_config).__name__}"
        )

    return normalized_sources


def download_dataset(
    dataset_name: str, output_dir: Path, splits: list[str] | None = None, config_name: str | None = None
) -> None:
    dataset_label = dataset_name if config_name is None else f"{dataset_name}/{config_name}"

    if splits:
        for split in splits:
            split_output_dir = output_dir / split
            print(f"Downloading {dataset_label} [{split}] to {split_output_dir}")
            dataset = load_dataset(dataset_name, name=config_name, split=split)
            dataset.save_to_disk(str(split_output_dir))
        return

    print(f"Downloading all splits for {dataset_label} to {output_dir}")
    dataset_dict = load_dataset(dataset_name, name=config_name)
    dataset_dict.save_to_disk(str(output_dir))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "dataset.yaml"
    raw_dir = repo_root / "data" / "raw"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print(f"Loaded config from {config_path}")

    for source_name, dataset_name, splits in iter_sources(config):
        source_dir = raw_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        try:
            download_dataset(dataset_name, source_dir, splits=splits)
        except ValueError as error:
            if "Config name is missing" not in str(error):
                raise

            config_names = get_dataset_config_names(dataset_name)
            if not config_names:
                raise

            print(
                f"Dataset {dataset_name} requires a config. Downloading all configs: "
                f"{', '.join(config_names)}"
            )
            for config_name in config_names:
                config_dir = source_dir / config_name
                config_dir.mkdir(parents=True, exist_ok=True)
                download_dataset(
                    dataset_name,
                    config_dir,
                    splits=splits,
                    config_name=config_name,
                )

    print("Finished downloading configured datasets.")


if __name__ == "__main__":
    main()
