from pathlib import Path

from datasets import load_dataset
import yaml


def iter_sources(config: dict) -> list[tuple[str, str, list[str] | None]]:
    sources = config.get("sources", {})

    if not isinstance(sources, dict):
        raise TypeError("'sources' must be a mapping in dataset.yaml")

    normalized_sources: list[tuple[str, str, list[str] | None]] = []

    for source_name, source_config in sources.items():
        if source_name == "n_dataset":
            continue

        if isinstance(source_config, dict):
            dataset_name = source_config["dataset"]
            splits = [split for split in source_config.get("splits", []) if split]
            normalized_sources.append((source_name, dataset_name, splits or None))
            continue

        if isinstance(source_config, str):
            normalized_sources.append((source_name, source_config, None))
            continue

        raise TypeError(
            f"Unsupported source config for '{source_name}': "
            f"expected dict or str, got {type(source_config).__name__}"
        )

    return normalized_sources


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

        if splits:
            for split in splits:
                output_dir = source_dir / split
                print(f"Downloading {dataset_name} [{split}] to {output_dir}")
                dataset = load_dataset(dataset_name, split=split)
                dataset.save_to_disk(str(output_dir))
            continue

        print(f"Downloading all splits for {dataset_name} to {source_dir}")
        dataset_dict = load_dataset(dataset_name)
        dataset_dict.save_to_disk(str(source_dir))

    print("Finished downloading configured datasets.")


if __name__ == "__main__":
    main()
