"""Deploy ArtSleuth to HuggingFace Spaces.

Copies relevant source directories to a staging folder, writes
requirements and metadata, then uploads via ``HfApi.upload_folder``.
Finally triggers a factory reboot to force a clean rebuild.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SPACE_ID = "ladyFaye1998/ArtSleuth"
LOCAL_ROOT = Path(__file__).resolve().parent

GRADIO_VERSION = "5.50.0"

REQUIREMENTS = f"""\
gradio[oauth]=={GRADIO_VERSION}
torch>=2.0
torchvision
transformers>=4.30
safetensors
huggingface-hub>=0.19
numpy
scipy
scikit-learn
Pillow
pydantic>=2.0
click
rich
ftfy
regex
"""

SPACE_README = f"""\
---
title: ArtSleuth
emoji: 🔍
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: {GRADIO_VERSION}
app_file: web/app.py
pinned: true
license: mit
---

# ArtSleuth — Computational Art Analysis

Upload a painting for style classification, artist attribution,
forgery screening, workshop decomposition, and temporal dating.

Powered by DINOv2 + CLIP dual-backbone fusion.

**GitHub:** [ladyFaye1998/ArtSleuth](https://github.com/ladyFaye1998/ArtSleuth)
"""

DIRS_TO_COPY = ["web", "artsleuth"]


def main() -> None:
    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()

        for d in DIRS_TO_COPY:
            src = LOCAL_ROOT / d
            dst = staging / d
            if src.exists():
                shutil.copytree(
                    src, dst,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", ".mypy_cache",
                    ),
                )
                print(f"  Staged {d}/")

        (staging / "requirements.txt").write_text(
            REQUIREMENTS.strip() + "\n", encoding="utf-8",
        )
        print("  Wrote requirements.txt")

        (staging / "README.md").write_text(
            SPACE_README, encoding="utf-8",
        )
        print("  Wrote README.md")

        print("Uploading to HuggingFace Space …")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message="Dark editorial UI redesign",
        )
        print("Upload complete.")

    print("Requesting factory reboot …")
    api.restart_space(SPACE_ID, factory_reboot=True)
    print("Factory reboot triggered — space will rebuild from scratch.")
    print(f"\nhttps://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
