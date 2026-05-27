# Darija LID Benchmark

This repository builds a Moroccan Darija language identification dataset from multiple public sources, normalizes them into one schema, removes duplicates, validates the final output, and supports publication to Hugging Face.

## Goal

The target task is 3-way text classification:

- `0` = `arabic_moroccan`
- `1` = `arabizi_moroccan`
- `2` = `other`

The `other` class includes non-Moroccan dialects, English translation rows, and the remaining non-target content.

## Final Dataset Schema

The final dataset contains these columns:

- `text`: normalized text
- `dialect`: normalized dialect name
- `src`: source dataset identifier, or multiple identifiers joined by `|`
- `has_latin_script`: whether the row contains at least one Latin character
- `label`: numeric class ID

Deduplication is done on normalized `text`.

## Source Datasets

Configured in [configs/dataset.yaml](/home/mohammed/Documents/AtlasIA/Darija_LID_Anootation_10k/darija-lid-benchmark/configs/dataset.yaml):

- `atlasia/Darija_LID_Anootation_10k`
- `atlasia/DODa-audio-dataset`
- `atlasia/Darija-LID`
- `UBC-NLP/alexandria`
- `atlasia/levantine_dialects`

## Pipeline

The pipeline has four stages:

1. `scripts/download_data.py`
- Downloads the configured raw datasets into `data/raw/`

2. `scripts/prepare_sources.py`
- Flattens each source into the shared schema
- Maps source-specific labels to the final classes
- Preserves source provenance

3. `scripts/build_dataset.py`
- Merges prepared sources
- Removes repeated rows
- Drops conflict groups where the same normalized text has inconsistent labels or dialects
- Writes the final dataset to `data/processed/`

4. `scripts/validate_dataset.py`
- Checks schema, label validity, duplicate removal, and script-flag consistency

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full data pipeline:

```bash
make data
```

Run validation:

```bash
make validate
```

Or run the steps manually:

```bash
python scripts/download_data.py
python scripts/prepare_sources.py
python scripts/build_dataset.py
python scripts/validate_dataset.py
```

## Outputs

Main outputs:

- `data/interim/prepared_sources/`: prepared per-source datasets
- `data/processed/final_dataset/`: Hugging Face disk-format dataset
- `data/processed/final_dataset.csv`: final CSV export
- `data/processed/build_summary.json`: build statistics
- `data/processed/validation_report.json`: validation results
- `data/processed/dropped_conflicts.json`: dropped conflict groups

Notebook outputs:

- [notebooks/data_understanding.ipynb](/home/mohammed/Documents/AtlasIA/Darija_LID_Anootation_10k/darija-lid-benchmark/notebooks/data_understanding.ipynb)
- `data/processed/plots/final_dataset_dashboard.png`

## Conflict Handling

If the same normalized text appears multiple times with conflicting labels or dialects, that conflict group is excluded from the final dataset instead of being resolved into one retained row.

The excluded groups are stored in:

- `data/processed/dropped_conflicts.json`

## Project Structure

- [src/darija_lid/load.py](/home/mohammed/Documents/AtlasIA/Darija_LID_Anootation_10k/darija-lid-benchmark/src/darija_lid/load.py): config loading and dataset helpers
- [src/darija_lid/clean.py](/home/mohammed/Documents/AtlasIA/Darija_LID_Anootation_10k/darija-lid-benchmark/src/darija_lid/clean.py): text normalization and script detection
- [src/darija_lid/sampling.py](/home/mohammed/Documents/AtlasIA/Darija_LID_Anootation_10k/darija-lid-benchmark/src/darija_lid/sampling.py): merge and deduplication logic
- [src/darija_lid/validation.py](/home/mohammed/Documents/AtlasIA/Darija_LID_Anootation_10k/darija-lid-benchmark/src/darija_lid/validation.py): validation logic
- [docs/label_policy.md](/home/mohammed/Documents/AtlasIA/Darija_LID_Anootation_10k/darija-lid-benchmark/docs/label_policy.md): label policy

## Publish

Push dataset artifacts to GitHub:

```bash
scripts/push_final_dataset.sh --message "Publish final dataset artifacts"
```

Publish to Hugging Face with dataset viewer and dataset card:

```bash
python scripts/publish_hf_dataset.py --repo-id YOUR_USERNAME/YOUR_DATASET_NAME
```

If you generated plots from the notebook first, they will also be uploaded and embedded in the Hugging Face dataset card.
