"""
See-through Easy — 手軽にレイヤー分解するためのラッパー

使い方:
  NF4: seethrough_easy.bat をダブルクリック（または画像をD&D）
  Full: seethrough_full.bat をダブルクリック（または画像をD&D）

  --full オプションで Full bf16 モード（VRAM ~14GB）
"""

import os
import sys
import subprocess
import time
import shutil

SEETHROUGH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# モデルは正方形入力固定（SDXLベース）。解像度を上げてもパディングが増えるだけ
RESOLUTION = 1280


def print_header():
    print()
    print("=" * 54)
    print("  See-through Easy")
    print("  アニメイラスト 1枚 → レイヤー分解 PSD")
    print("=" * 54)
    print()


def process_image(img_path, full_mode=False):
    """1枚の画像を処理する"""
    img_path = os.path.abspath(img_path)
    img_dir = os.path.dirname(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.join(img_dir, img_name)

    mode_str = "Full bf16" if full_mode else "NF4量子化"
    print(f"  入力:   {img_path}")
    print(f"  出力先: {output_dir}\\")
    print(f"  {mode_str} | 解像度: {RESOLUTION} | steps: 30 | seed: 42")
    print()
    print("-" * 54)
    print()

    start_time = time.time()

    if full_mode:
        # Full bf16（inference_psd.py）
        cmd = [
            sys.executable,
            os.path.join(SEETHROUGH_ROOT, "inference", "scripts", "inference_psd.py"),
            "--srcp", img_path,
            "--save_to_psd",
            "--save_dir", img_dir,
        ]
    else:
        # NF4量子化（inference_psd_quantized.py）
        cmd = [
            sys.executable,
            os.path.join(SEETHROUGH_ROOT, "inference", "scripts", "inference_psd_quantized.py"),
            "--srcp", img_path,
            "--save_to_psd",
            "--quant_mode", "nf4",
            "--resolution", str(RESOLUTION),
            "--save_dir", img_dir,
            "--no_group_offload",
        ]

    result = subprocess.run(cmd, cwd=SEETHROUGH_ROOT)

    if result.returncode != 0:
        print()
        print(f"  ✗ 推論が失敗しました (exit code: {result.returncode})")
        return False

    # PSD ファイルを出力フォルダに移動
    psd_patterns = [
        f"{img_name}.psd",
        f"{img_name}_depth.psd",
        f"{img_name}.psd.json",
    ]
    for filename in psd_patterns:
        src = os.path.join(img_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(output_dir, filename)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # サマリー表示
    print()
    print("-" * 54)
    print()

    if os.path.isdir(output_dir):
        files = os.listdir(output_dir)
        psd_files = [f for f in files if f.endswith(".psd")]
        png_files = [
            f for f in files
            if f.endswith(".png") and not f.startswith("src_")
        ]

        psd_path = os.path.join(output_dir, f"{img_name}.psd")
        psd_size_str = ""
        if os.path.exists(psd_path):
            size_mb = os.path.getsize(psd_path) / (1024 * 1024)
            psd_size_str = f" ({size_mb:.1f} MB)"

        print(f"  ✓ 完了！ ({minutes}分{seconds}秒)")
        print(f"  出力先:       {output_dir}\\")
        print(f"  PSD:          {len(psd_files)}個{psd_size_str}")
        print(f"  レイヤー画像: {len(png_files)}枚")
    else:
        print(f"  ✓ 完了！ ({minutes}分{seconds}秒)")

    print()
    return True


def main():
    # --full フラグの判定
    full_mode = "--full" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--full"]

    print_header()
    if full_mode:
        print("  *** Full bf16 モード (VRAM ~14GB) ***")
        print()

    # D&D またはコマンドライン引数があればそれを使う
    if args:
        img_path = args[0].strip().strip('"')
        if not os.path.isfile(img_path):
            print(f"  ✗ ファイルが見つかりません: {img_path}")
            return 1
        process_image(img_path, full_mode=full_mode)
        return 0

    # 対話モード（ループ）
    while True:
        print("-" * 54)
        raw = input("  画像パスを入力 (q で終了): ").strip().strip('"')

        if raw.lower() in ("q", "quit", "exit", ""):
            print()
            print("  おつかれさまでした！")
            break

        if not os.path.isfile(raw):
            print(f"  ✗ ファイルが見つかりません: {raw}")
            print()
            continue

        print()
        process_image(raw, full_mode=full_mode)
        print("=" * 54)
        print()


if __name__ == "__main__":
    sys.exit(main() or 0)
