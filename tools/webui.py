"""See-through WebUI — アニメイラスト レイヤー分解 Gradio インターフェース

New file added to See-through (https://github.com/shitagaki-lab/see-through).
Licensed under Apache License 2.0.
"""

import os
import sys
import re
import time
import subprocess
from pathlib import Path

import gradio as gr
from PIL import Image

# --- Paths ---
SEETHROUGH_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = SEETHROUGH_ROOT / "inference" / "scripts" / "inference_psd_quantized.py"
HF_CACHE_DIR = SEETHROUGH_ROOT / ".hf_cache"
OUTPUT_BASE = SEETHROUGH_ROOT / "workspace" / "webui_output"

SKIP_TAGS = {"src_img", "src_head", "reconstruction"}

LAYER_ORDER = [
    "front hair", "back hair", "head", "neck", "neckwear",
    "topwear", "handwear", "bottomwear", "legwear", "footwear",
    "tail", "wings", "objects",
    "headwear", "face", "irides", "eyebrow", "eyewhite",
    "eyelash", "eyewear", "ears", "earwear", "nose", "mouth",
]

# --- Stage detection for log parsing ---
STAGE_MARKERS = [
    ("Quantized inference:", "📋 推論設定"),
    ("Building LayerDiff", "🔨 LayerDiffパイプライン構築中..."),
    ("[NF4 fix]", "🔧 NF4テキストエンコーダ修正中..."),
    ("Running LayerDiff", "🎨 LayerDiff推論中 (body + head)..."),
    ("LayerDiff3D done", "✅ LayerDiff完了"),
    ("layerdiff pipeline freed", "♻️ VRAM解放中..."),
    ("Building Marigold", "🔨 Marigoldパイプライン構築中..."),
    ("Running Marigold", "🏔️ Marigold depth推論中..."),
    ("Marigold done", "✅ Marigold完了"),
    ("Running PSD assembly", "📦 PSD組み立て中..."),
    ("PSD assembly done", "✅ PSD完了"),
]


def _tag_sort_key(tag):
    try:
        return LAYER_ORDER.index(tag)
    except ValueError:
        return len(LAYER_ORDER)


def collect_layers(output_dir):
    """Collect layer PNGs as (filepath, label) tuples for the gallery."""
    if not os.path.isdir(output_dir):
        return []
    layers = []
    for f in os.listdir(output_dir):
        if not f.endswith(".png"):
            continue
        tag = f[:-4]
        if tag.endswith("_depth") or tag in SKIP_TAGS:
            continue
        layers.append((os.path.join(output_dir, f), tag))
    layers.sort(key=lambda x: _tag_sort_key(x[1]))
    return layers


def parse_log_status(log_path):
    """Parse log file tail to extract current stage and progress bar."""
    if not os.path.exists(log_path):
        return "⏳ 準備中..."

    try:
        size = os.path.getsize(log_path)
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(max(0, size - 6000))
            tail = f.read()
    except Exception:
        return "⏳ 処理中..."

    # Strip ANSI escape codes
    tail = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", tail)

    # Detect current stage (last match wins)
    current_stage = "⏳ 準備中..."
    for keyword, label in STAGE_MARKERS:
        if keyword in tail:
            current_stage = label

    # Extract latest tqdm / loading progress from \r-delimited segments
    progress_line = ""
    parts = tail.split("\r")
    for part in reversed(parts):
        part = part.strip()
        if not part:
            continue

        # tqdm diffusion steps: " 50%|█████     | 15/30 [01:38<01:40, 6.69s/it]"
        m = re.search(
            r"(\d+)%\|([^|]+)\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", part
        )
        if m:
            pct, bar, cur, total, timing = m.groups()
            progress_line = f"{pct}% |{bar.strip()}| {cur}/{total} [{timing}]"
            break

        # Loading weights / pipeline components
        m = re.search(r"Loading (\w+).*?(\d+)%\|([^|]+)\|\s*(\d+)/(\d+)", part)
        if m:
            what, pct, bar, cur, total = m.groups()
            label = "ウェイトロード" if what == "weights" else "パイプラインロード"
            progress_line = f"{label}: {pct}% |{bar.strip()}| {cur}/{total}"
            break

    if progress_line:
        return f"{current_stage}\n{progress_line}"
    return current_stage


def open_output_folder(output_path):
    """Open the output folder in Windows Explorer."""
    target = output_path if output_path and os.path.isdir(output_path) else str(OUTPUT_BASE)
    os.makedirs(target, exist_ok=True)
    os.startfile(target)


def get_vram_used_mb():
    """Get total VRAM used in MB via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            used, total = r.stdout.strip().split(", ")
            return int(used), int(total)
    except Exception:
        pass
    return 0, 0


def get_vram_display(baseline_mb):
    """Show VRAM usage: See-through delta + total."""
    used, total = get_vram_used_mb()
    if total == 0:
        return ""
    st_vram = max(0, used - baseline_mb)
    return f"🔍 See-through: ~{st_vram}MB | 全体: {used}/{total}MB"


def run_inference(image_path, mode_str, resolution, seed_val, tblr_split):
    """Run See-through inference. Generator that yields progressive updates."""
    if image_path is None:
        raise gr.Error("画像をアップロードしてください")

    # --- Setup ---
    seed_val = int(seed_val)
    resolution = int(resolution)
    # Snap to nearest 64
    resolution = max(512, min(2048, round(resolution / 64) * 64))
    is_nf4 = "NF4" in mode_str
    img_stem = Path(image_path).stem

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in img_stem)
    if not safe_name:
        safe_name = "image"

    run_id = f"{safe_name}_{int(time.time())}"
    save_dir = OUTPUT_BASE / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    input_path = save_dir / f"{safe_name}.png"
    Image.open(image_path).convert("RGBA").save(str(input_path))

    layer_dir = str(save_dir / safe_name)
    log_path = save_dir / "webui.log"

    yield [], str(save_dir), "⏳ 推論を開始します..."

    # --- Build command ---
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--srcp", str(input_path),
        "--save_to_psd",
        "--save_dir", str(save_dir),
        "--seed", str(seed_val),
        "--resolution", str(resolution),
        "--quant_mode", "nf4" if is_nf4 else "none",
        "--no_group_offload",
    ]
    if not tblr_split:
        cmd.append("--no_tblr_split")

    env = {**os.environ, "HF_HOME": str(HF_CACHE_DIR)}
    start_time = time.time()

    # Record baseline VRAM for tracking
    baseline_vram, _ = get_vram_used_mb()

    # --- Run subprocess ---
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(SEETHROUGH_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        while proc.poll() is None:
            time.sleep(2)
            layers = collect_layers(layer_dir)
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

            # Parse log for detailed status
            log_status = parse_log_status(str(log_path))
            vram = get_vram_display(baseline_vram)
            status_text = f"{log_status}\n⏱️ 経過時間: {elapsed_str}"
            if layers:
                status_text += f" | レイヤー: {len(layers)}枚"
            if vram:
                status_text += f"\n{vram}"

            yield layers, str(save_dir), status_text

    # --- Check result ---
    if proc.returncode != 0:
        err_tail = ""
        if log_path.exists():
            err_tail = log_path.read_text(encoding="utf-8")[-500:]
        raise gr.Error(f"推論失敗 (exit code: {proc.returncode})\n{err_tail}")

    # --- Final results ---
    gallery = collect_layers(layer_dir)
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    mode_label = "NF4" if is_nf4 else "Full bf16"

    # Count PSD files
    psd_count = sum(
        1 for f in os.listdir(str(save_dir))
        if f.endswith(".psd")
    ) if save_dir.exists() else 0

    # Read peak VRAM from stats.json
    import json as _json
    peak_vram_str = ""
    stats_path = save_dir / safe_name / "stats.json"
    if stats_path.exists():
        try:
            with open(stats_path, "r") as sf:
                stats_data = _json.load(sf)
            peak_gb = stats_data.get("peak_vram_gb", 0)
            peak_vram_str = f" | ピークVRAM: {peak_gb:.1f}GB"
        except Exception:
            pass

    status = (
        f"✅ 完了！ ({minutes}分{seconds}秒)\n"
        f"モード: {mode_label} | 解像度: {resolution} | "
        f"レイヤー: {len(gallery)}枚 | PSD: {psd_count}個{peak_vram_str}\n"
        f"📂 出力先: {save_dir}"
    )

    yield gallery, str(save_dir), status


# --- Custom CSS ---
CUSTOM_CSS = """
/* Checkerboard for transparent layer previews */
.gallery-item img,
div[data-testid="image"] img {
    background-image:
        linear-gradient(45deg, #ddd 25%, transparent 25%),
        linear-gradient(-45deg, #ddd 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #ddd 75%),
        linear-gradient(-45deg, transparent 75%, #ddd 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0;
    background-color: #e8e8ed;
}
.header-text { text-align: center; padding: 0.5rem 0; }
#status-box textarea {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.85rem;
    line-height: 1.4;
}
"""

# --- Theme: Anima-style warm orange on off-white ---
theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fef7f0", c100="#fde8d4", c200="#fbd0a8",
        c300="#f5b47a", c400="#f0a050", c500="#e8985a",
        c600="#d88a4e", c700="#c07838", c800="#a0632e",
        c900="#7a4c24", c950="#5a3818",
    ),
    secondary_hue="orange",
    neutral_hue=gr.themes.Color(
        c50="#f9f9f7", c100="#f3f3f0", c200="#ededea",
        c300="#d8d8d5", c400="#b0b0ad", c500="#9090a0",
        c600="#5c5c72", c700="#4a4a5e", c800="#2d2d3a",
        c900="#1e1e28", c950="#141420",
    ),
    font=[gr.themes.GoogleFont("Inter"), gr.themes.GoogleFont("Noto Sans JP"), "system-ui", "sans-serif"],
    font_mono=["Consolas", "Fira Code", "monospace"],
)

# --- Build UI ---
with gr.Blocks(title="See-through WebUI") as demo:

    # State: output folder path
    output_path_state = gr.State(value="")

    gr.Markdown(
        "# 🔍 See-through — アニメイラスト レイヤー分解\n"
        "アニメイラスト1枚から最大23レイヤーのセマンティック分解を行います。\n"
        "[GitHub](https://github.com/shitagaki-lab/see-through) | "
        "[論文](https://arxiv.org/abs/2602.03749) | "
        "SIGGRAPH 2026",
        elem_classes=["header-text"],
    )

    with gr.Row():
        # --- Left: Settings ---
        with gr.Column(scale=1, min_width=320):
            input_image = gr.Image(type="filepath", label="入力画像", height=350)

            with gr.Group():
                mode = gr.Radio(
                    choices=[
                        "NF4 量子化 (推奨・VRAM ~10GB)",
                        "Full bf16 (高品質・VRAM ~14GB)",
                    ],
                    value="NF4 量子化 (推奨・VRAM ~10GB)",
                    label="推論モード",
                )
                resolution = gr.Slider(
                    minimum=512, maximum=2048, step=64, value=1024,
                    label="解像度",
                    info="512: ~4GB / 768: ~5GB / 1024: ~7GB / 1280: ~9GB (NF4)",
                )
                with gr.Row():
                    seed = gr.Number(value=42, label="シード値", precision=0, minimum=0)
                    tblr_split = gr.Checkbox(
                        value=True, label="左右分割",
                        info="手・目などを左右に分離",
                    )

        # --- Right: Gallery + Actions ---
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="レイヤープレビュー", columns=4,
                height="auto", object_fit="contain",
            )

            run_btn = gr.Button("🚀 分解開始", variant="primary", size="lg")

            status = gr.Textbox(
                label="ステータス", interactive=False, lines=4,
                elem_id="status-box",
            )

            open_folder_btn = gr.Button("📂 出力フォルダを開く", size="sm")

    gr.Markdown(
        "---\n"
        "**出力先**: `workspace/webui_output/` | "
        "**推論時間の目安**: NF4 ~7分, Full ~7分 (RTX 3090)",
        elem_classes=["header-text"],
    )

    # --- Events ---
    run_btn.click(
        fn=run_inference,
        inputs=[input_image, mode, resolution, seed, tblr_split],
        outputs=[gallery, output_path_state, status],
    )

    open_folder_btn.click(
        fn=open_output_folder,
        inputs=[output_path_state],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        inbrowser=True, server_name="127.0.0.1",
        css=CUSTOM_CSS, theme=theme,
    )
