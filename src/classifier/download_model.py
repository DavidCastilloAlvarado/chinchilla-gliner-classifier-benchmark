"""Download GLiNER2 checkpoints into the project's temp/ folder.

Usage:
    uv run python src/classifier/download_model.py            # base (default)
    uv run python src/classifier/download_model.py multi      # gliner2.5-multi-v1
    uv run python src/classifier/download_model.py decide     # GLiNER2.5-multi-Decide
    uv run python src/classifier/download_model.py all        # all checkpoints
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "base": "fastino/gliner2-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
    "decide": "fastino/GLiNER2.5-multi-Decide",
}
TEMP_DIR = Path(__file__).resolve().parents[2] / "temp"


def download(name: str) -> None:
    repo = MODELS[name]
    local_dir = TEMP_DIR / repo.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo} -> {local_dir}")
    path = snapshot_download(repo_id=repo, local_dir=str(local_dir))
    print(f"Model ready at: {path}")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "base"
    if arg == "all":
        for name in MODELS:
            download(name)
    elif arg in MODELS:
        download(arg)
    else:
        raise SystemExit(f"Unknown model '{arg}'. Choose from: {', '.join(MODELS)} | all")


if __name__ == "__main__":
    main()
