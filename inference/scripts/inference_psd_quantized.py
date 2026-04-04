"""Quantized inference for See-through full pipeline (layerdiff body -> head -> marigold depth -> PSD).

Supports NF4 (default, 4-bit) and bf16 (baseline) modes. HF repos are auto-selected
based on quant_mode. Builds pipelines directly without using inference_utils global singletons.

New file added to See-through (https://github.com/shitagaki-lab/see-through).
Licensed under Apache License 2.0.

Usage (from repo root):
    python inference/scripts/inference_psd_quantized.py --srcp image.png --save_to_psd
    python inference/scripts/inference_psd_quantized.py --quant_mode none --no_group_offload
"""

import os.path as osp
import argparse
import sys
import os
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import json
import time
import cv2
import numpy as np
import torch
from PIL import Image

from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from modules.layerdiffuse.vae import TransparentVAE
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.marigold import MarigoldDepthPipeline
from utils.cv import center_square_pad_resize, smart_resize, img_alpha_blending
from utils.torch_utils import seed_everything
from utils.io_utils import json2dict, dict2json
from utils.inference_utils import further_extr
from utils.cv import validate_resolution


VALID_BODY_PARTS_V2 = [
    'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth',
    'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear',
    'tail', 'wings', 'objects'
]


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. In Google Colab, set Runtime > Change runtime type > Hardware accelerator > GPU."
        )


def is_quantized_module(module):
    if module is None:
        return False
    return any([
        getattr(module, 'is_quantized', False),
        getattr(module, 'is_loaded_in_4bit', False),
        getattr(module, 'is_loaded_in_8bit', False),
        getattr(module, 'quantization_method', None) is not None,
    ])


def move_module(module, *, device=None, dtype=None, label='module'):
    if module is None:
        return

    if is_quantized_module(module):
        print(f"  {label}: quantized module detected; leaving placement to transformers/bitsandbytes")
        return

    kwargs = {}
    if device is not None:
        kwargs['device'] = device
    if dtype is not None:
        kwargs['dtype'] = dtype
    if kwargs:
        module.to(**kwargs)


def build_layerdiff_pipeline(args):
    """Build the LayerDiff3D pipeline with appropriate quantization."""
    quant_mode = args.quant_mode

    if quant_mode == 'none':
        # bf16 baseline: load from original repo
        repo = args.repo_id_layerdiff
        trans_vae = TransparentVAE.from_pretrained(repo, subfolder='trans_vae')
        unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet')
        pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            repo, trans_vae=trans_vae, unet=unet, scheduler=None)
        if args.cpu_offload:
            move_module(pipeline.vae, dtype=torch.bfloat16, label='layerdiff.vae')
            move_module(pipeline.trans_vae, dtype=torch.bfloat16, label='layerdiff.trans_vae')
            move_module(pipeline.unet, dtype=torch.bfloat16, label='layerdiff.unet')
            move_module(pipeline.text_encoder, dtype=torch.bfloat16, label='layerdiff.text_encoder')
            move_module(pipeline.text_encoder_2, dtype=torch.bfloat16, label='layerdiff.text_encoder_2')
            pipeline.enable_model_cpu_offload()
        else:
            move_module(pipeline.vae, device='cuda', dtype=torch.bfloat16, label='layerdiff.vae')
            move_module(pipeline.trans_vae, device='cuda', dtype=torch.bfloat16, label='layerdiff.trans_vae')
            move_module(pipeline.unet, device='cuda', dtype=torch.bfloat16, label='layerdiff.unet')
            move_module(pipeline.text_encoder, device='cuda', dtype=torch.bfloat16, label='layerdiff.text_encoder')
            move_module(pipeline.text_encoder_2, device='cuda', dtype=torch.bfloat16, label='layerdiff.text_encoder_2')
            if getattr(args, 'group_offload', False):
                pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
        # Cache tag embeddings and unload text encoders to save VRAM
        pipeline.cache_tag_embeds()
    else:
        # NF4: load from pre-quantized repo (auto-selected by REPO_MAP)
        repo = args.repo_id_layerdiff
        unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet')

        trans_vae = TransparentVAE.from_pretrained(repo, subfolder='trans_vae')  # always bf16
        pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            repo, trans_vae=trans_vae, unet=unet, scheduler=None)

        if args.cpu_offload:
            # VAE + TransparentVAE to bf16; quantized components handled by bnb
            move_module(pipeline.vae, dtype=torch.bfloat16, label='layerdiff.vae')
            move_module(pipeline.trans_vae, dtype=torch.bfloat16, label='layerdiff.trans_vae')
            pipeline.enable_model_cpu_offload()
        else:
            move_module(pipeline.vae, device='cuda', dtype=torch.bfloat16, label='layerdiff.vae')
            move_module(pipeline.trans_vae, device='cuda', dtype=torch.bfloat16, label='layerdiff.trans_vae')
            # Don't manually .to(cuda) quantized components -- bnb handles device placement
            if getattr(args, 'group_offload', False):
                pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
        # NF4テキストエンコーダをdequantize（CUBLAS対策）
        # bitsandbytes NF4 matmulはCUBLAS初期化エラーを起こすことがある。
        # テキストエンコーダは一度だけ使用（タグ埋め込みキャッシュ）なので
        # bf16に戻しても実用上の問題はない。
        for enc_name in ['text_encoder', 'text_encoder_2']:
            enc = getattr(pipeline, enc_name, None)
            if enc is not None and hasattr(enc, 'dequantize'):
                print(f'  [NF4 fix] Dequantizing {enc_name} to bf16...')
                setattr(pipeline, enc_name,
                        enc.dequantize().to(device='cuda', dtype=torch.bfloat16))
        # Cache tag embeddings and unload text encoders to save VRAM
        pipeline.cache_tag_embeds()
        # キャッシュ後にテキストエンコーダ解放 → VRAM節約
        for enc_name in ['text_encoder', 'text_encoder_2']:
            if hasattr(pipeline, enc_name):
                delattr(pipeline, enc_name)
        torch.cuda.empty_cache()

    return pipeline


def build_marigold_pipeline(args):
    """Build the Marigold depth pipeline with appropriate quantization."""
    quant_mode = args.quant_mode

    if quant_mode == 'none':
        repo = args.repo_id_depth
        unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet')
        marigold_pipe = MarigoldDepthPipeline.from_pretrained(repo, unet=unet)
        if args.cpu_offload:
            move_module(marigold_pipe, dtype=torch.bfloat16, label='marigold.pipeline')
            marigold_pipe.enable_model_cpu_offload()
        else:
            move_module(marigold_pipe, device='cuda', dtype=torch.bfloat16, label='marigold.pipeline')
            if getattr(args, 'group_offload', False):
                marigold_pipe.enable_group_offload('cuda', num_blocks_per_group=1)
        marigold_pipe.cache_tag_embeds()
    else:
        # NF4: load from pre-quantized repo (auto-selected by REPO_MAP)
        repo = args.repo_id_depth
        unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet', torch_dtype=torch.bfloat16)

        marigold_pipe = MarigoldDepthPipeline.from_pretrained(repo, unet=unet, torch_dtype=torch.bfloat16)
        move_module(marigold_pipe.vae, device='cuda', label='marigold.vae')
        move_module(marigold_pipe.unet, device='cuda', label='marigold.unet')
        move_module(marigold_pipe.text_encoder, device='cuda', label='marigold.text_encoder')
        if getattr(args, 'group_offload', False):
            marigold_pipe.enable_group_offload('cuda', num_blocks_per_group=1)
        marigold_pipe.cache_tag_embeds()

    return marigold_pipe


def run_layerdiff(pipeline, imgp, save_dir, seed, num_inference_steps, resolution):
    """Run LayerDiff3D body + head passes. Replicates inference_utils.py v3 logic exactly."""
    saved = osp.join(save_dir, osp.splitext(osp.basename(imgp))[0])
    os.makedirs(saved, exist_ok=True)
    input_img = np.array(Image.open(imgp).convert('RGBA'))
    fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution, return_pad_info=True)
    scale = pad_size[0] / resolution
    Image.fromarray(fullpage).save(osp.join(saved, 'src_img.png'))

    rng = torch.Generator(device='cuda').manual_seed(seed)

    # Body pass
    body_tag_list = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
    pipeline_output = pipeline(
        strength=1.0,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        generator=rng,
        guidance_scale=1.0,
        prompt=body_tag_list,
        negative_prompt='',
        fullpage=fullpage,
        group_index=0
    )
    images = pipeline_output.images
    for rst, tag in zip(pipeline_output.images, body_tag_list):
        Image.fromarray(rst).save(osp.join(saved, f'{tag}.png'))
    head_img = images[2]

    # Head crop
    head_tag_list = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth']
    hx0, hy0, hw, hh = cv2.boundingRect(cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8)))

    hx = int(hx0 * scale) - pad_pos[0]
    hy = int(hy0 * scale) - pad_pos[1]
    hw = int(hw * scale)
    hh = int(hh * scale)

    def _crop_head(img, xywh):
        x, y, w, h = xywh
        ih, iw = img.shape[:2]

        # Asymmetric padding: wider upward to capture headwear (hats, horns, etc.)
        pad_x = int(w * 0.30)
        pad_y_up = int(h * 0.60)
        pad_y_down = int(h * 0.30)

        x1 = max(x - min(pad_x, x), 0)
        x2 = min(x + w + min(pad_x, iw - (x + w)), iw)
        y1 = max(y - min(pad_y_up, y), 0)
        y2 = min(y + h + min(pad_y_down, ih - (y + h)), ih)
        return img[y1: y2, x1: x2], (x1, y1, x2, y2)

    input_head, (hx1, hy1, hx2, hy2) = _crop_head(input_img, [hx, hy, hw, hh])
    hx1 = int(hx1 / scale + pad_pos[0] / scale)
    hy1 = int(hy1 / scale + pad_pos[1] / scale)
    ih, iw = input_head.shape[:2]
    input_head, pad_size, pad_pos = center_square_pad_resize(input_head, resolution, return_pad_info=True)
    Image.fromarray(input_head).save(osp.join(saved, 'src_head.png'))

    # Head pass
    pipeline_output = pipeline(
        strength=1.0,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        generator=rng,
        guidance_scale=1.0,
        prompt=head_tag_list,
        negative_prompt='',
        fullpage=input_head,
        group_index=1
    )
    canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)

    py1, py2, px1, px2 = (np.array([pad_pos[1], pad_pos[1] + ih, pad_pos[0], pad_pos[0] + iw]) / scale).astype(np.int64)

    scale_size = (int(pad_size[0] / scale), int(pad_size[1] / scale))

    for rst, tag in zip(pipeline_output.images, head_tag_list):
        rst = smart_resize(rst, scale_size)[py1: py2, px1: px2]
        full = canvas.copy()
        full[hy1: hy1 + rst.shape[0], hx1: hx1 + rst.shape[1]] = rst
        Image.fromarray(full).save(osp.join(saved, f'{tag}.png'))


def run_marigold(marigold_pipe, srcp, save_dir, seed, resolution_depth):
    """Run Marigold depth estimation. Matches inference_utils.apply_marigold logic.

    Uses resolution_depth to control Marigold inference resolution. If different from
    source image size, images are resized before depth prediction and depth maps are
    resized back after. All frames processed together (no chunking).
    """
    srcname = osp.basename(osp.splitext(srcp)[0])
    saved = osp.join(save_dir, srcname)

    # Read source image to get actual size (matches inference_utils approach)
    src_img_p = osp.join(saved, 'src_img.png')
    fullpage = np.array(Image.open(src_img_p).convert('RGBA'))
    src_h, src_w = fullpage.shape[:2]

    if isinstance(resolution_depth, int) and resolution_depth == -1:
        resolution_depth = [src_h, src_w]
    resolution_depth = validate_resolution(resolution_depth)
    src_rescaled = resolution_depth[0] != src_h or resolution_depth[1] != src_w

    img_list = []
    exist_list = []
    empty_array = np.zeros((src_h, src_w, 4), dtype=np.uint8)
    blended_alpha = np.zeros((src_h, src_w), dtype=np.float32)

    compose_list = {'eyes': ['eyewhite', 'irides', 'eyelash', 'eyebrow'], 'hair': ['back hair', 'front hair']}
    for tag in VALID_BODY_PARTS_V2:
        tagp = osp.join(saved, f'{tag}.png')
        if osp.exists(tagp):
            exist_list.append(True)
            tag_arr = np.array(Image.open(tagp))
            tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
            img_list.append(tag_arr)
        else:
            img_list.append(empty_array)
            exist_list.append(False)

    compose_dict = {}
    for c, clist in compose_list.items():
        imlist = []
        taglist = []
        for tag in clist:
            p = osp.join(saved, tag + '.png')
            if osp.exists(p):
                tag_arr = np.array(Image.open(p))
                tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                imlist.append(tag_arr)
                taglist.append(tag)
        if len(imlist) > 0:
            img = img_alpha_blending(imlist, premultiplied=False)
            img_list[VALID_BODY_PARTS_V2.index(c)] = img
            compose_dict[c] = {'taglist': taglist, 'imlist': imlist}

    for img in img_list:
        blended_alpha += img[..., -1].astype(np.float32) / 255

    blended_alpha = np.clip(blended_alpha, 0, 1) * 255
    blended_alpha = blended_alpha.astype(np.uint8)
    fullpage[..., -1] = blended_alpha
    img_list.append(fullpage)

    # Resize to depth resolution if needed
    img_list_input = img_list
    if src_rescaled:
        img_list_input = [smart_resize(img, resolution_depth) for img in img_list]

    seed_everything(seed)
    pipe_out = marigold_pipe(color_map=None, img_list=img_list_input)
    depth_pred = pipe_out.depth_tensor
    depth_pred = depth_pred.to(device='cpu', dtype=torch.float32).numpy()

    # Resize depth back to source resolution if needed
    if src_rescaled:
        depth_pred = [smart_resize(d, (src_h, src_w)) for d in depth_pred]

    drawables = [{'img': img, 'depth': depth} for img, depth in zip(img_list, depth_pred)]
    drawables = drawables[:-1]
    blended = img_alpha_blending(drawables, premultiplied=False)

    infop = osp.join(saved, 'info.json')
    if osp.exists(infop):
        info = json2dict(infop)
    else:
        info = {'parts': {}}

    parts = info['parts']
    for ii, depth in enumerate(depth_pred[:-1]):
        depth = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        tag = VALID_BODY_PARTS_V2[ii]
        if tag in compose_dict:
            mask = blended_alpha > 256
            for t, im in zip(compose_dict[tag]['taglist'][::-1], compose_dict[tag]['imlist'][::-1]):
                mask_local = im[..., -1] > 15
                mask_invis = np.bitwise_and(mask, mask_local)
                depth_local = np.full((src_h, src_w), fill_value=255, dtype=np.uint8)
                depth_local[mask_local] = depth[mask_local]
                if np.any(mask_invis):
                    depth_local[mask_invis] = np.median(depth[np.bitwise_and(mask_local, np.bitwise_not(mask_invis))])
                mask = np.bitwise_or(mask, mask_local)

                parts_info = parts.get(t, {})
                Image.fromarray(depth_local).save(osp.join(saved, f'{t}_depth.png'))
                parts[t] = parts_info
            continue

        parts_info = parts.get(tag, {})
        Image.fromarray(depth).save(osp.join(saved, f'{tag}_depth.png'))
        parts[tag] = parts_info

    dict2json(info, infop)
    Image.fromarray(blended).save(osp.join(saved, 'reconstruction.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Quantized inference: LayerDiff body+head -> Marigold depth -> PSD"
    )
    parser.add_argument('--srcp', type=str, default='assets/test_image.png', help='input image')
    parser.add_argument('--save_dir', type=str, default='workspace/layerdiff_output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--save_to_psd', action='store_true')
    parser.add_argument('--tblr_split', action='store_true', default=True,
                        help='split parts (handwear, eyes, etc) into left-right components (default: on)')
    parser.add_argument('--no_tblr_split', dest='tblr_split', action='store_false',
                        help='disable left-right split')
    parser.add_argument('--quant_mode', type=str, default='nf4', choices=['nf4', 'none'],
                        help='quantization mode: nf4 (default, 4-bit) or none (bf16 baseline)')
    parser.add_argument('--repo_id_layerdiff', type=str, default=None,
                        help='Override LayerDiff3D HF repo (auto-selected based on quant_mode)')
    parser.add_argument('--repo_id_depth', type=str, default=None,
                        help='Override Marigold3D HF repo (auto-selected based on quant_mode)')
    parser.add_argument('--cpu_offload', action='store_true', default=False,
                        help='enable model CPU offload (default: on)')
    parser.add_argument('--no_cpu_offload', action='store_false', dest='cpu_offload',
                        help='disable model CPU offload')
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--resolution_depth', type=int, default=720,
                        help='Marigold depth inference resolution (default 720; -1 to match layerdiff resolution)')
    parser.add_argument('--group_offload', action='store_true', default=True,
                        help='Enable group offload to reduce peak VRAM (default: on)')
    parser.add_argument('--no_group_offload', action='store_false', dest='group_offload',
                        help='Disable group offload for faster inference on high-VRAM GPUs')
    args = parser.parse_args()

    # Auto-select HF repos based on quant_mode
    REPO_MAP = {
        'nf4': {
            'layerdiff': '24yearsold/seethroughv0.0.2_layerdiff3d_nf4',
            'depth': '24yearsold/seethroughv0.0.1_marigold_nf4',
        },
        'none': {
            'layerdiff': 'layerdifforg/seethroughv0.0.2_layerdiff3d',
            'depth': '24yearsold/seethroughv0.0.1_marigold',
        },
    }
    defaults = REPO_MAP[args.quant_mode]
    if args.repo_id_layerdiff is None:
        args.repo_id_layerdiff = defaults['layerdiff']
    if args.repo_id_depth is None:
        args.repo_id_depth = defaults['depth']

    srcp = args.srcp
    seed = args.seed
    resolution = args.resolution
    num_inference_steps = args.num_inference_steps
    save_dir = args.save_dir
    srcname = osp.basename(osp.splitext(srcp)[0])
    saved = osp.join(save_dir, srcname)

    print(f"Quantized inference: quant_mode={args.quant_mode}, cpu_offload={args.cpu_offload}")
    print(f"  Source image: {srcp}")
    print(f"  Save dir: {save_dir}")
    print(f"  Resolution: {resolution}, Steps: {num_inference_steps}, Seed: {seed}")

    require_cuda()
    torch.cuda.reset_peak_memory_stats()
    total_t0 = time.time()

    # --- LayerDiff ---
    print('\nBuilding LayerDiff3D pipeline...')
    seed_everything(seed)
    pipeline = build_layerdiff_pipeline(args)

    print('Running LayerDiff3D (body + head)...')
    layerdiff_t0 = time.time()
    run_layerdiff(pipeline, srcp, save_dir, seed, num_inference_steps, resolution)
    layerdiff_time = time.time() - layerdiff_t0
    print(f'  LayerDiff3D done in {layerdiff_time:.1f}s')

    # Free layerdiff pipeline before loading marigold
    del pipeline
    torch.cuda.empty_cache()

    # --- Marigold ---
    print('\nBuilding Marigold depth pipeline...')
    marigold_pipe = build_marigold_pipeline(args)

    print('Running Marigold depth...')
    marigold_t0 = time.time()
    run_marigold(marigold_pipe, srcp, save_dir, seed, resolution_depth=args.resolution_depth)
    marigold_time = time.time() - marigold_t0
    print(f'  Marigold done in {marigold_time:.1f}s')

    # Free marigold pipeline before PSD assembly
    del marigold_pipe
    torch.cuda.empty_cache()

    # --- PSD assembly ---
    print('\nRunning PSD assembly...')
    psd_t0 = time.time()
    further_extr(saved, rotate=False, save_to_psd=args.save_to_psd, tblr_split=args.tblr_split)
    psd_time = time.time() - psd_t0
    print(f'  PSD assembly done in {psd_time:.1f}s')

    total_time = time.time() - total_t0

    # --- Stats ---
    stats = {
        'quant_mode': args.quant_mode,
        'peak_vram_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'layerdiff_time_s': layerdiff_time,
        'marigold_time_s': marigold_time,
        'psd_time_s': psd_time,
        'total_time_s': total_time,
    }
    print(f'\n{"="*60}')
    print(json.dumps(stats, indent=2))
    print(f'{"="*60}')
    with open(osp.join(saved, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Stats saved to {osp.join(saved, "stats.json")}')
