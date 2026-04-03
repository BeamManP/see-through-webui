"""
See-through Easy — 手軽にレイヤー分解するためのラッパー

使い方:
  1. seethrough_easy.bat をダブルクリック
  2. 画像パスを入力してエンター
  3. 画像と同じフォルダに結果が保存される

  または: 画像ファイルを seethrough_easy.bat にドラッグ＆ドロップ
"""

import os
import sys
import subprocess
import time
import shutil
import math

SEETHROUGH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ピクセル予算（この面積に収まるよう解像度を計算）
TARGET_PIXELS = 1280 * 1280
# 解像度の上限（VRAM保護）
MAX_RESOLUTION = 1792
# 解像度は64の倍数にする（SDXL要件）
RESOLUTION_STEP = 64


def calc_resolution(img_w, img_h):
    """入力画像のアスペクト比に合わせて最適な正方形解像度を計算する。

    モデルは正方形入力を受け取り、画像は center_square_pad で
    正方形に引き伸ばされる。アスペクト比が1:1から離れるほど
    パディング領域が増えて解像度が無駄になる。

    この関数は「画像コンテンツ部分のピクセル数 ≈ TARGET_PIXELS」
    になるように正方形サイズ S を計算する。

    公式: ratio = min(w,h) / max(w,h)
           コンテンツ面積 = ratio * S²
           ratio * S² = TARGET_PIXELS
           S = sqrt(TARGET_PIXELS / ratio)
    """
    ratio = min(img_w, img_h) / max(img_w, img_h)
    s = math.sqrt(TARGET_PIXELS / ratio)
    # 64の倍数に丸める
    s = int(round(s / RESOLUTION_STEP) * RESOLUTION_STEP)
    # 上限クランプ
    s = min(s, MAX_RESOLUTION)
    # 下限（最低でも元の解像度は確保）
    s = max(s, RESOLUTION_STEP)
    return s


def print_header():
    print()
    print("=" * 54)
    print("  See-through Easy")
    print("  アニメイラスト 1枚 → レイヤー分解 PSD")
    print("=" * 54)
    print()


def process_image(img_path):
    """1枚の画像を処理する"""
    img_path = os.path.abspath(img_path)
    img_dir = os.path.dirname(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.join(img_dir, img_name)

    # 画像サイズ取得 → アスペクト比に応じた解像度計算
    from PIL import Image
    with Image.open(img_path) as im:
        img_w, img_h = im.size
    resolution = calc_resolution(img_w, img_h)
    ratio_str = f"{img_w}x{img_h}"

    print(f"  入力:   {img_path} ({ratio_str})")
    print(f"  出力先: {output_dir}\\")
    print(f"  NF4量子化 | 左右分割: ON | 解像度: {resolution} (自動) | steps: 30 | seed: 42")
    print()
    print("-" * 54)
    print()

    start_time = time.time()

    # 推論実行（inference_psd_quantized.py NF4モード）
    cmd = [
        sys.executable,
        os.path.join(SEETHROUGH_ROOT, "inference", "scripts", "inference_psd_quantized.py"),
        "--srcp", img_path,
        "--save_to_psd",
        "--quant_mode", "nf4",
        "--resolution", str(resolution),
        "--save_dir", img_dir,
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
    print_header()

    # D&D またはコマンドライン引数があればそれを使う
    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip().strip('"')
        if not os.path.isfile(img_path):
            print(f"  ✗ ファイルが見つかりません: {img_path}")
            return 1
        process_image(img_path)
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
        process_image(raw)
        print("=" * 54)
        print()


if __name__ == "__main__":
    sys.exit(main() or 0)
