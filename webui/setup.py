"""See-through WebUI — セットアップスクリプト

install.bat から呼ばれる。venv 内の Python で実行される前提。
pip install、CUDA 検証、モデルダウンロードを行う。
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_ROOT / "install.log"
HF_CACHE = REPO_ROOT / ".hf_cache"

PYTORCH_PACKAGES = [
    "torch==2.8.0+cu128",
    "torchvision==0.23.0+cu128",
    "torchaudio==2.8.0+cu128",
]
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"

MODELS = [
    ("LayerDiff NF4", "24yearsold/seethroughv0.0.2_layerdiff3d_nf4"),
    ("Marigold NF4", "24yearsold/seethroughv0.0.1_marigold_nf4"),
]


# ---------- Helpers ----------
def log(msg: str):
    """画面とログファイル両方に書く"""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_pip(args: list[str], desc: str) -> bool:
    """pip コマンドを実行。画面に出力を表示しつつ成否を返す。"""
    cmd = [sys.executable, "-m", "pip"] + args
    log(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        log(f"  [FAILED] {desc}")
        return False
    log(f"  [OK] {desc}")
    return True


def header(step: int, total: int, title: str):
    log(f"\n[{step}/{total}] {title}")
    log("=" * 50)


# ---------- Steps ----------
def step_pip_upgrade():
    header(1, 5, "Upgrading pip")
    run_pip(["install", "--upgrade", "pip", "--quiet"], "pip upgrade")


def step_pytorch() -> bool:
    header(2, 5, "Installing PyTorch + CUDA 12.8")
    log("  (first time: several minutes)")
    args = ["install"] + PYTORCH_PACKAGES + ["--index-url", PYTORCH_INDEX]
    return run_pip(args, "PyTorch")


def step_dependencies() -> bool:
    header(3, 5, "Installing dependencies")

    # common + annotators (editable)
    log("  common / annotators ...")
    if not run_pip(["install", "-e", "./common", "-e", "./annotators"], "common/annotators"):
        return False

    # webui requirements
    log("  webui requirements ...")
    req_file = REPO_ROOT / "webui" / "requirements.txt"
    if not run_pip(["install", "-r", str(req_file)], "webui requirements"):
        return False

    # assets folder
    assets_dst = REPO_ROOT / "assets"
    assets_src = REPO_ROOT / "common" / "assets"
    if not assets_dst.exists() and assets_src.exists():
        log("  Copying assets ...")
        shutil.copytree(assets_src, assets_dst)
        log("  [OK] assets")

    return True


def step_cuda_verify() -> bool:
    header(4, 5, "Verifying CUDA")
    try:
        import torch
        if not torch.cuda.is_available():
            log("  [FAILED] CUDA is not available.")
            log("  Make sure you have an NVIDIA GPU and an up-to-date driver.")
            log("  https://www.nvidia.com/drivers")
            return False
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        log(f"  CUDA: {cuda_ver}  GPU: {gpu_name}")
        log("  [OK]")
        return True
    except Exception as e:
        log(f"  [FAILED] {e}")
        return False


def step_models() -> bool:
    header(5, 5, "Downloading NF4 models")
    log("  (first time: ~3GB download)")

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log("  [FAILED] huggingface_hub not installed")
        return False

    cache_dir = str(HF_CACHE)
    for name, repo_id in MODELS:
        log(f"  {name} ...")
        try:
            snapshot_download(
                repo_id,
                cache_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            log(f"  [FAILED] {name}: {e}")
            return False

    log("  [OK] All models downloaded.")
    return True


# ---------- Main ----------
def main():
    # Init log
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"setup.py started\n")
        f.write(f"Python: {sys.executable}\n")
        f.write(f"Version: {sys.version}\n")
        f.write(f"{'='*50}\n")

    print()
    print("=" * 50)
    print("  See-through WebUI Setup")
    print("=" * 50)

    # Step 1: pip
    step_pip_upgrade()

    # Step 2: PyTorch
    if not step_pytorch():
        log("\n[ERROR] PyTorch install failed.")
        log("Check your network connection and try again.")
        sys.exit(1)

    # Step 3: Dependencies
    if not step_dependencies():
        log("\n[ERROR] Dependency install failed.")
        log("Check install.log for details.")
        sys.exit(1)

    # Step 4: CUDA
    if not step_cuda_verify():
        log("\n[ERROR] CUDA verification failed.")
        sys.exit(1)

    # Step 5: Models
    if not step_models():
        log("\n[ERROR] Model download failed.")
        log("You can retry by running install.bat again.")
        sys.exit(1)

    # Done
    print()
    print("=" * 50)
    print()
    print("  Install complete!")
    print()
    print("  To start: double-click run.bat")
    print()
    print("=" * 50)
    print()
    log("\n[SUCCESS] Install complete.")


if __name__ == "__main__":
    main()
