"""See-through WebUI — リリースzip作成スクリプト

配布に必要なファイルだけを集めてzipにパッケージングする。
Usage: python webui/package_release.py [--output release.zip]
"""

import argparse
import os
import zipfile
from pathlib import Path

# リポジトリルートからの相対パス
INCLUDE_FILES = [
    # --- Root ---
    "install.bat",
    "run.bat",
    "README.md",
    "LICENSE",
    ".gitignore",

    # --- WebUI ---
    "webui/README.md",
    "webui/requirements.txt",
    "webui/package_release.py",

    # --- Tools ---
    "tools/webui.py",

    # --- Inference ---
    "inference/__init__.py",
    "inference/scripts/__init__.py",
    "inference/scripts/inference_psd_quantized.py",
]

INCLUDE_DIRS = [
    # --- common (core model code + utilities) ---
    "common/modules",
    "common/utils",
    "common/live2d",
    "common/assets",

    # --- annotators (only lama_inpainter needed for inference) ---
    "annotators/lama_inpainter",
]

INCLUDE_DIR_FILES = [
    # --- Package configs ---
    "common/pyproject.toml",
    "annotators/__init__.py",
    "annotators/pyproject.toml",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".egg-info",
    ".DS_Store",
    "Thumbs.db",
]


def should_exclude(path: str) -> bool:
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path:
            return True
    return False


def package_release(repo_root: Path, output: Path):
    repo_root = repo_root.resolve()
    count = 0

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        # Individual files
        for rel in INCLUDE_FILES:
            src = repo_root / rel
            if src.exists():
                zf.write(src, f"see-through-webui/{rel}")
                count += 1
            else:
                print(f"  [SKIP] {rel} (not found)")

        # Individual files in directories
        for rel in INCLUDE_DIR_FILES:
            src = repo_root / rel
            if src.exists():
                zf.write(src, f"see-through-webui/{rel}")
                count += 1

        # Directories (recursive)
        for dir_rel in INCLUDE_DIRS:
            dir_path = repo_root / dir_rel
            if not dir_path.exists():
                print(f"  [SKIP] {dir_rel}/ (not found)")
                continue
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    full = Path(root) / f
                    rel = full.relative_to(repo_root)
                    if should_exclude(str(rel)):
                        continue
                    zf.write(full, f"see-through-webui/{rel}")
                    count += 1

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"\n  パッケージ作成完了: {output}")
    print(f"  ファイル数: {count}")
    print(f"  サイズ: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="See-through WebUI release packager")
    parser.add_argument("--output", type=str, default="see-through-webui.zip")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    package_release(repo_root, Path(args.output))
