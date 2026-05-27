#!/usr/bin/env bash

set -euo pipefail

REMOTE="origin"
BRANCH=""
COMMIT_MESSAGE="Add final dataset artifacts"
INCLUDE_HF_DIR="false"

usage() {
  cat <<'EOF'
Usage: scripts/push_final_dataset.sh [options]

Stages the final dataset artifacts, creates a commit, and pushes to GitHub.

Options:
  --remote <name>         Git remote to push to. Default: origin
  --branch <name>         Branch to push to. Default: current branch
  --message <message>     Commit message. Default: "Add final dataset artifacts"
  --include-hf-dir        Also add data/processed/final_dataset/ (Arrow dataset)
  --help                  Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --message)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --include-hf-dir)
      INCLUDE_HF_DIR="true"
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${BRANCH}" ]]; then
  BRANCH="$(git branch --show-current)"
fi

CSV_PATH="data/processed/final_dataset.csv"
BUILD_SUMMARY_PATH="data/processed/build_summary.json"
VALIDATION_REPORT_PATH="data/processed/validation_report.json"
HF_DIR_PATH="data/processed/final_dataset"

for required_path in "${CSV_PATH}" "${BUILD_SUMMARY_PATH}" "${VALIDATION_REPORT_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required artifact: ${required_path}" >&2
    echo "Run scripts/build_dataset.py and scripts/validate_dataset.py first." >&2
    exit 1
  fi
done

git add -f "${CSV_PATH}" "${BUILD_SUMMARY_PATH}" "${VALIDATION_REPORT_PATH}"

if [[ "${INCLUDE_HF_DIR}" == "true" ]]; then
  if [[ ! -d "${HF_DIR_PATH}" ]]; then
    echo "Missing Hugging Face dataset directory: ${HF_DIR_PATH}" >&2
    exit 1
  fi
  git add -f "${HF_DIR_PATH}"
fi

if git diff --cached --quiet; then
  echo "No staged changes for final dataset artifacts."
  exit 0
fi

git commit -m "${COMMIT_MESSAGE}"
git push "${REMOTE}" "${BRANCH}"

echo "Pushed final dataset artifacts to ${REMOTE}/${BRANCH}"
