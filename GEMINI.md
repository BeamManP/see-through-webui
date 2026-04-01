# See-through — 手順書

アニメイラスト1枚から最大23レイヤーのセマンティック分解を行うツール。
SIGGRAPH 2026 採択。

## リポジトリ
- GitHub: https://github.com/shitagaki-lab/see-through
- 論文: https://arxiv.org/abs/2602.03749
- デモ: https://huggingface.co/spaces/24yearsold/see-through-demo

## 環境

| 項目 | 値 |
|------|-----|
| conda環境名 | `see_through` |
| Python | 3.12 |
| PyTorch | 2.8.0 + CUDA 12.8 |
| HFキャッシュ | `F:\seethrough\.hf_cache` |
| 出力先 | `workspace/layerdiff_output/` |

## 基本コマンド

### 推論（1枚絵 → PSD）

```powershell
conda activate see_through
$env:HF_HOME = "F:\seethrough\.hf_cache"
python inference/scripts/inference_psd.py --srcp <画像パス> --save_to_psd
```

- 初回はHuggingFaceからモデルをDL（合計 ~10GB）
- 処理時間: RTX 3090で約7分/枚
- 出力: `workspace/layerdiff_output/<画像名>.psd` と `<画像名>_depth.psd`

### フォルダ一括処理

```powershell
python inference/scripts/inference_psd.py --srcp <フォルダパス> --save_to_psd
```

### 深度ベース追加分割

```powershell
python inference/scripts/heuristic_partseg.py seg_wdepth --srcp workspace/layerdiff_output/<名前>.psd --target_tags handwear
```

### 左右分割

```powershell
python inference/scripts/heuristic_partseg.py seg_wlr --srcp workspace/layerdiff_output/<名前>_wdepth.psd --target_tags handwear-1
```

## 出力レイヤー（V3 / 23パーツ）

### ボディグループ（group_index=0）
front hair, back hair, head, neck, neckwear, topwear, handwear, bottomwear, legwear, footwear, tail, wings, objects

### ヘッドグループ（group_index=1）
headwear, face, irides, eyebrow, eyewhite, eyelash, eyewear, ears, earwear, nose, mouth

## 既知の問題と対策

### VRAM OOM（修正済み）
- **問題**: LayerDiff（SDXLベース、VRAM ~14GB）が終わった後、パイプラインがVRAMに残ったままMarigold（VRAM ~4GB）をロードしてOOM
- **対策**: `inference_psd.py` を修正し、LayerDiff完了後に `layerdiff_pipeline = None` + `torch.cuda.empty_cache()` でVRAM解放
- **VRAM使用量**: ピーク約14GB（LayerDiff推論時）。24GB GPUなら余裕あり

### Windows固有
- `ln -sf` の代わりに `Copy-Item` で `common/assets` → `assets` にコピー済み
- `HF_HOME` 環境変数を毎回セットする必要あり（PowerShellセッション内のみ有効）
- detectron2 / mmdet（オプション）はWindowsビルドが不安定。メインパイプラインには不要

### シンボリックリンク
- 管理者権限がないため `mklink /D` が使えなかった
- `assets/` フォルダは `common/assets/` のコピーで代用

## モデル一覧

| モデル | HuggingFace | サイズ |
|--------|-------------|--------|
| LayerDiff 3D (SDXL) | layerdifforg/seethroughv0.0.2_layerdiff3d | ~10GB |
| Marigold Depth | 24yearsold/seethroughv0.0.1_marigold | ~850MB |

## メモ

- トレーニングスクリプトは 2026/04/12 リリース予定
- ComfyUI統合: https://github.com/jtydhr88/ComfyUI-See-through
- ライセンス: Apache 2.0
