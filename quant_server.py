"""
Standalone local quantization tool for ComfyUI models.

Starts a small local web server (reachable only from your own PC,
127.0.0.1) with a browser UI. There you enter the path to a bf16 model
file and choose the quantization format, and the script runs `ctq`
(convert_to_quant) in the background with the matching parameters.

For INT8 tensor-wise/ConvRot the marker fix is applied automatically
afterwards (ctq writes two extra fields into the `.comfy_quant` marker that
ComfyUI doesn't know, so it discards the whole marker -> "unet unexpected"
warning). This step trims the marker to exactly the fields that the official
Comfy-Org file uses as well.

IMPORTANT: This is a local application, not a public web tool. It reads and
writes files on YOUR computer and runs a program (ctq) on YOUR computer. It
cannot and must not run as a hosted website -- only locally, with the same
Python that your ComfyUI uses.

Start (with the ComfyUI Python):
    "C:\\ComfyUI_windows_portable\\python_embeded\\python.exe" -m pip install flask
    "C:\\ComfyUI_windows_portable\\python_embeded\\python.exe" quant_server.py
On Linux/macOS: ./start_server.sh, or python quant_server.py.

Then open in the browser: http://127.0.0.1:8877
"""

import atexit
import datetime
import json
import os
import re
import shutil
import signal
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

# static_folder=None: /static/ is served by a custom route below (without browser caching),
# so changes to style.css/app.js are visible right after a reload.
app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# Configuration: model folders, ComfyUI installation, ctq path override.
# Stored permanently in config.json next to this script so nothing has to be
# hardcoded in the code and changes survive a server restart. On the very first
# start there are no model folders yet -- Settings then opens automatically
# with a hint.
# ---------------------------------------------------------------------------

APP_VERSION = "1.3"
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CONFIG_PATH = SCRIPT_DIR / "config.json"

DEFAULT_MODEL_ROOTS = []


def _guess_comfyui_root(python_exe=None):
    """Guesses the ComfyUI installation from the Python interpreter. Windows portable:
    python_embeded and ComfyUI sit next to each other. Linux/macOS: usually a venv lives
    in or next to the ComfyUI folder (ComfyUI/venv/bin/python or venv/bin/python next to
    ComfyUI). Only a suggestion -- the user can change it in Settings."""
    exe = Path(python_exe or sys.executable)
    candidate = exe.parent.parent / "ComfyUI"
    if candidate.is_dir():
        return str(candidate)
    for ancestor in list(exe.parents)[:4]:
        if (ancestor / "custom_nodes").is_dir() and (ancestor / "main.py").is_file():
            return str(ancestor)
        if (ancestor / "ComfyUI" / "custom_nodes").is_dir():
            return str(ancestor / "ComfyUI")
    return ""


def load_config():
    """Returns (config, existed_before) -- existed_before says whether config.json
    was already there before this start (for the one-time first-run display)."""
    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            data.setdefault("model_roots", DEFAULT_MODEL_ROOTS)
            data.setdefault("comfyui_root", _guess_comfyui_root())
            data.setdefault("ctq_path_override", None)
            return data, True
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "model_roots": DEFAULT_MODEL_ROOTS,
        "comfyui_root": _guess_comfyui_root(),
        "ctq_path_override": None,
    }, False


def save_config():
    CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")


CONFIG, CONFIG_EXISTED_AT_START = load_config()
if not CONFIG_EXISTED_AT_START:
    save_config()

# ---------------------------------------------------------------------------
# Locating ctq
# ---------------------------------------------------------------------------


def find_ctq_executable(override=None):
    """Looks for ctq(.exe). A manual override (from Settings) takes precedence;
    otherwise the Scripts folder next to the current Python interpreter is searched
    -- that fits any venv/embedded Python installation, no matter where exactly
    ComfyUI lives, so we don't have to hardcode the path.
    """
    if override and Path(override).is_file():
        return override
    py_dir = Path(sys.executable).parent
    candidates = [
        py_dir / "Scripts" / "ctq.exe",
        py_dir / "Scripts" / "ctq",
        py_dir / "ctq.exe",
        py_dir / "ctq",
        py_dir.parent / "Scripts" / "ctq.exe",
        py_dir.parent / "bin" / "ctq",
    ]
    for c in candidates:
        if c.is_file() and (os.name == "nt" or os.access(c, os.X_OK)):
            return str(c)
    if os.name != "nt":
        # Linux/macOS: "pip install --user" puts ctq in ~/.local/bin, not next to python.
        found = shutil.which("ctq")
        if found:
            return found
        local = Path.home() / ".local" / "bin" / "ctq"
        if local.is_file() and os.access(local, os.X_OK):
            return str(local)
    return None


CTQ_EXE = find_ctq_executable(CONFIG.get("ctq_path_override"))
INT8FAST_REPO_URL = "https://github.com/BobJohnson24/ComfyUI-INT8-Fast"
INT8FAST_DIR_NAME = "ComfyUI-INT8-Fast"
# Our own repo: the update check asks it for the latest release.
GFLAVA_REPO = "Freeman79680/GFlava-Quant"
GFLAVA_RELEASES_URL = f"https://github.com/{GFLAVA_REPO}/releases"

# ---------------------------------------------------------------------------
# Quantization formats
#
# Each entry: the ctq flags we actually saw confirmed in the docs/examples of
# convert_to_quant. "verified" marks what we actually loaded and checked
# end-to-end in ComfyUI in our own test (currently only INT8 ConvRot). The other
# formats are called correctly according to ctq's own docs, but we haven't
# verified their marker format ourselves -- so the marker fix stays off for them.
# ---------------------------------------------------------------------------

FORMAT_PRESETS = {
    "int8_convrot": {
        "label": "INT8 ConvRot (empfohlen, von uns verifiziert)",
        "label_en": "INT8 ConvRot (recommended, verified by us)",
        "description": "8-Bit-Ganzzahlen pro Gewicht, zusaetzlich vorher mit einer Hadamard-Rotation "
                        "gedreht, die Ausreisser in den Zahlen glaettet. Laedt nativ in ComfyUI, keine "
                        "Zusatz-Node noetig. Nach allem, was wir gesehen haben, das beste Verhaeltnis "
                        "aus Dateigroesse, Geschwindigkeit und Bildqualitaet -- unsere Standard-Empfehlung.",
        "description_en": "8-bit integers per weight, additionally rotated beforehand with a Hadamard "
                           "rotation that smooths out outliers in the numbers. Loads natively in ComfyUI, "
                           "no extra node needed. From what we've seen, the best balance of file size, "
                           "speed and image quality -- our default recommendation.",
        "flags": ["--int8", "--scaling_mode", "row", "--convrot", "--simple"],
        "needs_groupsize": True,
        "marker_fix": True,
        "verified": True,
    },
    "int8_tensorwise": {
        "label": "INT8 Tensor-Wise (ohne ConvRot)",
        "label_en": "INT8 Tensor-Wise (without ConvRot)",
        "description": "Wie INT8 ConvRot, nur ohne die vorherige Rotation. Etwas schneller zu erzeugen, "
                        "gilt in Community-Vergleichen aber tendenziell als etwas ungenauer als ConvRot.",
        "description_en": "Like INT8 ConvRot, just without the prior rotation. A bit faster to produce, "
                           "but tends to be seen as slightly less accurate than ConvRot in community "
                           "comparisons.",
        "flags": ["--int8", "--scaling_mode", "tensor", "--simple"],
        "needs_groupsize": False,
        "marker_fix": True,
        "verified": False,
    },
    "int8_block": {
        "label": "INT8 Block-Wise (gelernte Rundung, langsamer)",
        "label_en": "INT8 Block-Wise (learned rounding, slower)",
        "description": "8-Bit mit gelernter Rundung (SVD-Optimierung) pro kleinem Zahlenblock statt pro "
                        "ganzer Zeile oder Tensor. Deutlich langsamer beim Erzeugen, kann bei manchen "
                        "Modellen aber genauer sein.",
        "description_en": "8-bit with learned rounding (SVD optimization) per small block of numbers "
                           "instead of per full row or tensor. Considerably slower to produce, but can be "
                           "more accurate on some models.",
        # --scaling_mode block is required, otherwise ctq ignores --block_size entirely and
        # quantizes tensor-wise (confirmed via `ctq --help-experimental`).
        "flags": ["--int8", "--scaling_mode", "block"],
        "needs_groupsize": False,
        "needs_blocksize": True,
        "marker_fix": False,
        "verified": False,
    },
    "fp8": {
        "label": "FP8 (Standard-Format, Ada/Hopper+)",
        "label_en": "FP8 (default format, Ada/Hopper+)",
        "description": "Gleitkommazahlen mit 8 Bit statt Ganzzahlen, keine Rotation noetig. Laeuft nativ "
                        "schnell auf neueren Nvidia-Karten (RTX 40xx/Hopper und neuer).",
        "description_en": "8-bit floating point numbers instead of integers, no rotation needed. Runs "
                           "natively fast on newer Nvidia cards (RTX 40xx/Hopper and newer).",
        "flags": ["--simple"],
        "needs_groupsize": False,
        "marker_fix": False,
        "verified": False,
    },
    "nvfp4": {
        "label": "NVFP4 (4-Bit, Blackwell, braucht comfy-kitchen)",
        "label_en": "NVFP4 (4-bit, Blackwell, needs comfy-kitchen)",
        "description": "Nur 4 Bit pro Zahl -- deutlich kleinere Datei, aber sichtbar mehr Qualitaetsverlust "
                        "als 8-Bit-Formate. Volle Geschwindigkeit nur auf Blackwell-Karten (RTX 50xx), "
                        "braucht zusaetzlich das Paket comfy-kitchen.",
        "description_en": "Only 4 bits per number -- a much smaller file, but noticeably more quality "
                           "loss than 8-bit formats. Full speed only on Blackwell cards (RTX 50xx), "
                           "additionally needs the comfy-kitchen package.",
        "flags": ["--nvfp4"],
        "needs_groupsize": False,
        "marker_fix": False,
        "verified": False,
    },
    "mxfp8": {
        "label": "MXFP8 (Blackwell)",
        "label_en": "MXFP8 (Blackwell)",
        "description": "8-Bit-Gleitkomma-Variante mit blockweiser Skalierung (Microscaling), speziell fuer "
                        "Blackwell-Karten optimiert.",
        "description_en": "An 8-bit floating point variant with block-wise scaling (microscaling), "
                           "specifically optimized for Blackwell cards.",
        "flags": ["--mxfp8"],
        "needs_groupsize": False,
        "marker_fix": False,
        "verified": False,
    },
}

# Layer exclusion presets. "regex" = custom value for --exclude-layers
# (with "verified": true we tested it end-to-end ourselves, otherwise it is a
# pattern derived from a known/publicly documented architecture that we have
# NOT compared against an official file). "builtin" = one of ctq's own model
# flags (see `ctq -hf`); those are appended as a standalone flag, not as an
# --exclude-layers value. "chip" = short name for the quick-select tiles at the
# top of the form (optional).
EXCLUDE_PRESETS = {
    "qwen21_custom": {
        "label": "Qwen-Image-2.1 (Single-Stream) -- empfohlen fuer deine Modelle",
        "label_en": "Qwen-Image-2.1 (single-stream) -- recommended for your models",
        "chip": "Qwen-Image 2.1",
        "kind": "regex",
        "value": r"(img_in|txt_in|modulation|norm_out|proj_out|time_text_embed|norm_q|norm_k)",
        "verified": True,
    },
    "flux1_custom": {
        "label": "Flux.1 (dev/schnell) -- Community-Regex, von uns nicht verifiziert",
        "label_en": "Flux.1 (dev/schnell) -- community regex, not verified by us",
        "chip": "Flux.1 (dev/schnell)",
        "kind": "regex",
        "value": r"(img_in|txt_in|time_in|vector_in|guidance_in|final_layer|modulation|norm)",
        "verified": False,
    },
    "sdxl_custom": {
        "label": "SDXL / Illustrious (SDXL-basiert) -- Community-Regex, von uns nicht verifiziert",
        "label_en": "SDXL / Illustrious (SDXL-based) -- community regex, not verified by us",
        "chip": "SDXL / Illustrious",
        "kind": "regex",
        "value": r"(input_blocks\.0\.0|out\.2|time_embed|label_emb)",
        "verified": False,
    },
    "none": {"label": "Kein Ausschluss", "label_en": "No exclusion", "kind": "none", "value": None},
    "custom": {"label": "Eigenes Regex eingeben", "label_en": "Enter custom regex", "kind": "custom", "value": None},
    # -- ctq's own model presets (from `ctq --help-filters`, as of our installation). --
    # "filter_desc" is the description that ctq itself prints for this preset, verbatim --
    # not interpreted by us, so nothing here is guessed. It is already English (ctq's
    # own CLI output) and is therefore shown unchanged in both languages.
    "ctq_qwen": {
        "label": "ctq --qwen (aelteres Qwen Image, Doppel-Strom)",
        "label_en": "ctq --qwen (older Qwen Image, dual-stream)",
        "chip": "Qwen (Dual-Stream)", "kind": "builtin", "value": "--qwen",
        "filter_desc": "Qwen Image: skip added norms, keep time_text_embed high-precision",
    },
    "ctq_zimage": {
        "label": "ctq --zimage", "label_en": "ctq --zimage",
        "chip": "Z-Image", "kind": "builtin", "value": "--zimage",
        "filter_desc": "Z-Image: skip cap_embedder/norms, keep x_embedder/final high-precision",
    },
    "ctq_zimage_refiner": {
        "label": "ctq --zimage_refiner", "label_en": "ctq --zimage_refiner",
        "chip": "Z-Image Refiner", "kind": "builtin", "value": "--zimage_refiner",
        "filter_desc": "Z-Image Refiner: keep context/noise refiner high-precision",
    },
    "ctq_boogu": {
        "label": "ctq --boogu", "label_en": "ctq --boogu",
        "chip": "Boogu", "kind": "builtin", "value": "--boogu",
        "filter_desc": "Boogu: keep image_index_embedding, ref_image_patch_embedder, time_caption_embed, "
                        "x_embedder high-precision",
    },
    "ctq_flux2": {
        "label": "ctq --flux2", "label_en": "ctq --flux2",
        "chip": "Flux.2", "kind": "builtin", "value": "--flux2",
        "filter_desc": "Flux.2: keep modulation/guidance/time/final layers high-precision",
    },
    "ctq_anima": {
        "label": "ctq --anima -- von dir Ende-zu-Ende getestet (Anima 2.9B, 40 Bloecke)",
        "label_en": "ctq --anima -- end-to-end tested by you (Anima 2.9B, 40 blocks)",
        "chip": "Anima", "kind": "builtin", "value": "--anima", "verified": True,
        "filter_desc": "Anima diffusion model: keep first blocks, adaln_modulation, final/embedding layers "
                        "high-precision",
    },
    "ctq_lens": {
        "label": "ctq --lens", "label_en": "ctq --lens",
        "chip": "LENS", "kind": "builtin", "value": "--lens",
        "filter_desc": "LENS diffusion model: keep time_text_embed, img_in, norm_out, proj_out, some mod "
                        "layers high-precision",
    },
    "ctq_krea2": {
        "label": "ctq --krea2", "label_en": "ctq --krea2",
        "chip": "Krea2", "kind": "builtin", "value": "--krea2",
        "filter_desc": "Krea2: keep firs, las, tml, txtfusion, last.modulation, tpro layers high-precision",
        "filter_desc_note_de": "(Wortlaut so von ctq selbst, teils abgekuerzt)",
        "filter_desc_note_en": "(wording exactly as given by ctq itself, partly abbreviated)",
    },
    "ctq_ideogram4": {
        "label": "ctq --ideogram4", "label_en": "ctq --ideogram4",
        "chip": "Ideogram4", "kind": "builtin", "value": "--ideogram4",
        "filter_desc": "Ideogram4: keep embed_image_indicator, t_embedding, adaln_proj, final_layer, "
                        "input_proj layers high-precision",
    },
    "ctq_distillation_large": {
        "label": "ctq --distillation_large (Chroma gross)",
        "label_en": "ctq --distillation_large (Chroma large)",
        "chip": "Chroma", "kind": "builtin",
        "value": "--distillation_large",
        "filter_desc": "Chroma/distilled (large): keep distilled_guidance, final, img/txt_in high-precision",
    },
    "ctq_distillation_small": {
        "label": "ctq --distillation_small (Chroma klein)",
        "label_en": "ctq --distillation_small (Chroma small)",
        "chip": "Chroma (klein)", "kind": "builtin", "value": "--distillation_small",
        "filter_desc": "Chroma/distilled (small): keep only distilled_guidance high-precision",
    },
    "ctq_nerf_large": {
        "label": "ctq --nerf_large", "label_en": "ctq --nerf_large",
        "chip": "NeRF (gross)", "kind": "builtin", "value": "--nerf_large",
        "filter_desc": "NeRF (large): keep nerf_blocks, distilled_guidance, txt_in high-precision",
    },
    "ctq_nerf_small": {
        "label": "ctq --nerf_small", "label_en": "ctq --nerf_small",
        "chip": "NeRF (klein)", "kind": "builtin", "value": "--nerf_small",
        "filter_desc": "NeRF (small): keep nerf_blocks, distilled_guidance high-precision",
    },
    "ctq_radiance": {
        "label": "ctq --radiance", "label_en": "ctq --radiance",
        "chip": "Radiance", "kind": "builtin", "value": "--radiance",
        "filter_desc": "Radiance model: keep img_in_patch, nerf_final_layer high-precision",
    },
    "ctq_wan": {
        "label": "ctq --wan (Video)", "label_en": "ctq --wan (video)",
        "chip": "Wan (Video)", "kind": "builtin", "value": "--wan",
        "filter_desc": "WAN video model: skip embeddings, encoders, head",
    },
    "ctq_hunyuan": {
        "label": "ctq --hunyuan (Video)", "label_en": "ctq --hunyuan (video)",
        "chip": "HunyuanVideo", "kind": "builtin", "value": "--hunyuan",
        "filter_desc": "Hunyuan Video 1.5: skip layernorm, attn norms, vision_in",
    },
    "ctq_minimaxh3": {
        "label": "ctq --minimaxh3 (Video)", "label_en": "ctq --minimaxh3 (video)",
        "chip": "MiniMax H3", "kind": "builtin", "value": "--minimaxh3",
        "filter_desc": "MiniMax H3: keep patch/condition/final/time and token-refiner layers high-precision",
    },
    "ctq_ltxv2": {
        "label": "ctq --ltxv2 (Video)", "label_en": "ctq --ltxv2 (video)",
        "chip": "LTXv2", "kind": "builtin", "value": "--ltxv2",
        "filter_desc": "LTXv2: keep some transformer blocks high-precision and exclude vae and vocoder",
    },
    "ctq_gemma4": {
        "label": "ctq --gemma4 (Text-Encoder)", "label_en": "ctq --gemma4 (text encoder)",
        "chip": "Gemma4", "kind": "builtin", "value": "--gemma4",
        "filter_desc": "Gemma4 text/multimodal model: skip audio, per_layer_input_gate, per_layer_projection, "
                        "vision, multi_modal_projector",
    },
    "ctq_qwen_vlm": {
        "label": "ctq --qwen_vlm (Qwen3-VL Text-Encoder)",
        "label_en": "ctq --qwen_vlm (Qwen3-VL text encoder)",
        "chip": "Qwen3-VL", "kind": "builtin", "value": "--qwen_vlm",
        "filter_desc": "Qwen VLM family: skip first/last language layers, embeddings, MTP, and the full "
                        "visual encoder",
    },
    "ctq_t5xxl": {
        "label": "ctq --t5xxl (Text-Encoder)", "label_en": "ctq --t5xxl (text encoder)",
        "chip": "T5-XXL", "kind": "builtin", "value": "--t5xxl",
        "filter_desc": "T5-XXL text encoder: skip norms/biases, remove decoder layers",
    },
    "ctq_mistral": {
        "label": "ctq --mistral (Text-Encoder)", "label_en": "ctq --mistral (text encoder)",
        "chip": "Mistral", "kind": "builtin", "value": "--mistral",
        "filter_desc": "Mistral text encoder exclusions",
    },
    "ctq_visual": {
        "label": "ctq --visual (Vision-Encoder)", "label_en": "ctq --visual (vision encoder)",
        "chip": "Vision-Encoder", "kind": "builtin", "value": "--visual",
        "filter_desc": "Visual encoder: skip MLP layers (down/up/gate proj)",
    },
    "ctq_generic_text": {
        "label": "ctq --generic_text (allg. Text-Encoder)",
        "label_en": "ctq --generic_text (generic text encoder)",
        "chip": "Text-Encoder (allg.)", "kind": "builtin", "value": "--generic_text",
        "filter_desc": "Generic text encoder: skip MLP layers (down/up/gate proj)",
    },
}

# Which fields we keep in the .comfy_quant marker, depending on the "format"
# value stored IN the marker itself. Only for formats that we compared against
# an official Comfy-Org file ourselves.
MARKER_ALLOWED_FIELDS = {
    "int8_tensorwise": ["format", "convrot", "convrot_groupsize"],
}

# ---------------------------------------------------------------------------
# Logs: every quantization run is saved completely and permanently under logs/
# (it survives a server restart too), so a run can be traced afterwards.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model folders for the selection dropdown. They now come from CONFIG (see
# Settings in the browser), no longer hardcoded. They are searched recursively
# for .safetensors; the dropdown shows for every model which folder (and, if
# applicable, which subfolder) it comes from.
# ---------------------------------------------------------------------------


def _find_safetensors(base):
    """All .safetensors files under base. Follows symlinks to folders (on Linux/macOS model
    folders are often symlinked; Path.rglob doesn't follow those) and ignores the
    case of the extension (Linux distinguishes it, Windows doesn't)."""
    found = []
    seen = set()  # real paths of already visited folders -- prevents endless loops through symlink cycles
    for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        for name in filenames:
            if name.lower().endswith(".safetensors"):
                found.append(Path(dirpath) / name)
    return found


def scan_models():
    entries = []
    for root in CONFIG.get("model_roots", []):
        base = Path(root["path"])
        if not base.is_dir():
            continue
        try:
            found = _find_safetensors(base)
        except OSError:
            continue
        for p in found:
            try:
                if not p.is_file():
                    continue
                rel_dir = p.parent.relative_to(base)
                size_mb = round(p.stat().st_size / (1024 * 1024), 1)
            except OSError:
                continue
            subdir = "" if str(rel_dir) == "." else str(rel_dir).replace("\\", "/")
            entries.append({
                "path": str(p),
                "filename": p.name,
                "subdir": subdir,
                "root_key": root["key"],
                "root_label": root["label"],
                "size_mb": size_mb,
            })
    entries.sort(key=lambda e: (e["root_label"], e["subdir"], e["filename"].lower()))
    return entries


# ---------------------------------------------------------------------------
# Model type detection: reads only the JSON header of a .safetensors file
# (no tensor data is loaded, so it is very fast even for large models) and
# checks the tensor names against signatures that either come directly from
# ctq's own installed MODEL_FILTERS constants (real ground truth -- see
# convert_to_quant/constants.py) or were verified by us against a real file
# (Qwen-Image 2.1 single-stream: no official ctq constant, since ctq's own
# --qwen means the OLDER dual-stream variant). Every rule requires ALL "require"
# substrings and NONE of the "forbid" substrings in the tensor names -- better
# no detection than a wrong one. Architectures without a truly distinguishable
# signature (currently only LENS -- too similar to Qwen dual-stream, shares
# "img_mod"/"time_text_embed"/"img_in" without a unique feature of its own)
# are deliberately NOT included.
# ---------------------------------------------------------------------------


def _get_ctq_constants():
    try:
        from convert_to_quant import constants as ctq_constants
        return ctq_constants
    except Exception:
        return None


def build_model_detection_registry():
    registry = [
        # Verified against a real qwen_image_2.1_bf16.safetensors: 32
        # blocks with transformer_blocks.N.img_mlp.*, a single
        # modulation.1.weight, txt_in.text_norm.weight -- no img_mod/
        # txt_mod as in ctq's older dual-stream variant (--qwen).
        {"preset_key": "qwen21_custom", "require": ["img_mlp", "txt_in.text_norm"], "forbid": ["img_mod", "txt_mod"]},
        # Verified against a real flux1-dev-bnb-nf4.safetensors:
        # double_blocks/single_blocks with img_mod.lin/txt_mod.lin per block --
        # unlike Flux.2 (double_stream_modulation_img/txt, no img_mod)
        # and unlike ctq's Qwen dual-stream (transformer_blocks instead of
        # double_blocks/single_blocks).
        {"preset_key": "flux1_custom", "require": ["double_blocks", "single_blocks", "img_mod", "txt_mod"],
         "forbid": ["stream_modulation"]},
        # Verified against a real SDXL/Illustrious file: U-Net layout
        # (input_blocks/middle_block/output_blocks/label_emb) does not occur in any
        # DiT architecture (Flux/Qwen/Anima/Z-Image/...) -- a completely different
        # layout, hence especially safe to recognize.
        {"preset_key": "sdxl_custom", "require": ["input_blocks", "middle_block", "output_blocks", "label_emb"],
         "forbid": []},
    ]

    if _get_ctq_constants() is not None:
        # From here on: substrings 1:1 from ctq's own MODEL_FILTERS/*_LAYER_KEYNAMES
        # (see constants.py of the installed convert_to_quant version) --
        # not guessed by us, but what ctq itself considers characteristic for each
        # architecture. Additionally verified manually against real Anima-3.8B and
        # Flux.2-Klein files.
        registry += [
            {"preset_key": "ctq_anima", "require": ["llm_adapter", "adaln_modulation"], "forbid": []},
            {"preset_key": "ctq_flux2", "require": ["stream_modulation"], "forbid": []},
            {"preset_key": "ctq_qwen", "require": ["img_mod"],
             "forbid": ["llm_adapter", "stream_modulation", "double_blocks"]},
            {"preset_key": "ctq_zimage_refiner", "require": ["context_refiner", "noise_refiner"], "forbid": []},
            {"preset_key": "ctq_zimage", "require": ["cap_embedder", "adaLN_modulation"],
             "forbid": ["context_refiner"]},
            {"preset_key": "ctq_wan", "require": ["casual_audio_encoder"], "forbid": []},
            {"preset_key": "ctq_hunyuan", "require": ["vision_in.proj", "cond_type_embedding"], "forbid": []},
            {"preset_key": "ctq_krea2", "require": ["txtfusion"], "forbid": []},
            {"preset_key": "ctq_boogu", "require": ["ref_image_patch_embedder"], "forbid": []},
            {"preset_key": "ctq_ideogram4", "require": ["embed_image_indicator"], "forbid": []},
            {"preset_key": "ctq_radiance", "require": ["img_in_patch", "nerf_final_layer"], "forbid": []},
            {"preset_key": "ctq_nerf_large", "require": ["nerf_blocks", "nerf_image_embedder"], "forbid": []},
            {"preset_key": "ctq_nerf_small", "require": ["nerf_blocks"], "forbid": ["nerf_image_embedder"]},
            {"preset_key": "ctq_distillation_large", "require": ["distilled_guidance_layer", "img_in", "txt_in"],
             "forbid": ["nerf_blocks"]},
            {"preset_key": "ctq_distillation_small", "require": ["distilled_guidance_layer"],
             "forbid": ["nerf_blocks", "img_in"]},
            {"preset_key": "ctq_minimaxh3", "require": ["audio_patch_proj", "token_refiner"], "forbid": []},
            {"preset_key": "ctq_ltxv2", "require": ["scale_shift_table", "patchify_proj"], "forbid": []},
            {"preset_key": "ctq_gemma4", "require": ["per_layer_input_gate", "per_layer_projection"], "forbid": []},
            {"preset_key": "ctq_qwen_vlm", "require": ["mtp.", "visual."], "forbid": []},
        ]
    return registry


MODEL_DETECTION_REGISTRY = build_model_detection_registry()


def read_safetensors_header(path):
    """Reads only the JSON header of a .safetensors file (8-byte length +
    header), no tensor data -- fast even for very large models."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        if header_size <= 0 or header_size > 200_000_000:
            raise ValueError("Ungueltiger oder zu grosser Header.")
        return json.loads(f.read(header_size))


def read_safetensors_keys(path):
    return [k for k in read_safetensors_header(path).keys() if k != "__metadata__"]


def analyze_safetensors_header(header):
    """For the size estimate in the UI: how many bytes sit in 16-bit and 32-bit
    weights with at least 2 dimensions (those are the tensors ctq can quantize at all).
    A rough approximation only."""
    b16 = b32 = 0
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape") or []
        offsets = info.get("data_offsets") or [0, 0]
        if len(shape) < 2 or len(offsets) != 2:
            continue
        size = max(0, int(offsets[1]) - int(offsets[0]))
        dtype = info.get("dtype")
        if dtype in ("BF16", "F16"):
            b16 += size
        elif dtype == "F32":
            b32 += size
    return {"bytes_16bit_2d": b16, "bytes_32bit_2d": b32}


def detect_model_type(tensor_keys):
    def has(sub):
        return any(sub in k for k in tensor_keys)

    for rule in MODEL_DETECTION_REGISTRY:
        if all(has(sub) for sub in rule["require"]) and not any(has(sub) for sub in rule["forbid"]):
            return rule["preset_key"]
    return None


JOBS = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()

# Queue: jobs run strictly one after another (to avoid GPU/RAM conflicts), but
# any number can be queued. A single worker thread picks up the next job as soon
# as the previous one has finished. The queue lives in the server, not in the
# browser -- a closed or reloaded tab therefore loses nothing.
JOB_QUEUE = []                      # job_ids in order, protected by JOBS_LOCK
QUEUE_COND = threading.Condition(JOBS_LOCK)
ACTIVE_STATES = ("queued", "starting", "running")
FINAL_STATES = ("done", "error", "cancelled")
MAX_FINISHED_JOBS = 30              # this many finished jobs stay in memory for display
_QUEUE_WORKER = {"thread": None}


# ---------------------------------------------------------------------------
# Marker fix (step 16 of our manual procedure, now automated)
# ---------------------------------------------------------------------------


def fix_markers(path, log):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    fixed = 0
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.endswith(".comfy_quant"):
                raw = t.numpy().tobytes().rstrip(b"\x00")
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    tensors[k] = t
                    continue
                allowed = MARKER_ALLOWED_FIELDS.get(data.get("format"))
                if allowed:
                    minimal = {kk: data[kk] for kk in allowed if kk in data}
                    if minimal != data:
                        new_bytes = json.dumps(minimal).encode("utf-8")
                        t = torch.frombuffer(bytearray(new_bytes), dtype=torch.uint8).clone()
                        fixed += 1
            tensors[k] = t

    tmp_path = path + ".marker_fix_tmp"
    save_file(tensors, tmp_path, metadata={"format": "pt"})
    os.replace(tmp_path, path)
    log(f"Marker bereinigt: {fixed} von {sum(1 for k in tensors if k.endswith('.comfy_quant'))} comfy_quant-Tensoren")


# ---------------------------------------------------------------------------
# Low-memory mode: according to `ctq --help`, ctq's --low-memory explicitly concerns
# system RAM ("streaming tensor loading to reduce RAM usage, recommended
# for models >50% of available RAM"), NOT the GPU's VRAM. We checked this in
# the installed ctq source (utils/memory_efficient_loader.py): without this flag
# ctq loads all tensors up front into a Python dict in RAM, with the flag it reads
# them one by one from disk. The actual GPU processing (converters/*.py) moves only
# one weight tensor at a time to the GPU either way and then calls
# torch.cuda.empty_cache() -- that happens automatically and independently of
# --low-memory, so there is nothing VRAM-related to detect. What we can detect
# automatically (and what matches ctq's own recommendation) is whether the input
# file takes up more than half of the available system RAM.
# ---------------------------------------------------------------------------

LOW_MEMORY_RAM_RATIO = 0.5


def _parse_meminfo(text):
    """'MemAvailable' from the contents of /proc/meminfo (Linux) in bytes, otherwise None."""
    m = re.search(r"^MemAvailable:\s+(\d+)\s*kB", text, re.MULTILINE)
    return int(m.group(1)) * 1024 if m else None


def _parse_vm_stat(text):
    """Available RAM from the output of `vm_stat` (macOS) in bytes, otherwise None.
    Free, inactive and speculative pages count as available (the system releases the
    inactive ones immediately when needed)."""
    size = re.search(r"page size of (\d+) bytes", text)
    if not size:
        return None
    pages = 0
    found = False
    for label in ("free", "inactive", "speculative"):
        m = re.search(rf"^Pages {label}:\s+(\d+)", text, re.MULTILINE)
        if m:
            pages += int(m.group(1))
            found = True
    return pages * int(size.group(1)) if found else None


def _get_available_ram_posix():
    """Linux and macOS. psutil first (present in every ComfyUI environment, since ComfyUI
    needs it itself), otherwise /proc/meminfo or vm_stat."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        return _parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        pass
    if sys.platform == "darwin":
        code, out = _run_cmd(["vm_stat"], timeout=5)
        if code == 0:
            return _parse_vm_stat(out)
    return None


def get_available_ram_bytes():
    """Available physical system RAM in bytes, or None if it cannot be
    determined."""
    if os.name != "nt":
        try:
            return _get_available_ram_posix()
        except Exception:
            return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return int(stat.ullAvailPhys)
    except Exception:
        return None


def resolve_low_memory_flag(mode, input_path, log):
    """Decides whether --low-memory is passed to ctq. mode is "auto"
    (RAM-based detection), "on" (always) or "off" (never)."""
    if mode == "on":
        log("Low-memory-Modus: manuell erzwungen (immer an).")
        return True
    if mode == "off":
        log("Low-memory-Modus: manuell deaktiviert (immer aus).")
        return False

    try:
        file_size = os.path.getsize(input_path)
    except OSError:
        file_size = None
    available_ram = get_available_ram_bytes()

    if not file_size or not available_ram:
        log("Low-memory-Modus (automatisch): Dateigroesse oder verfuegbarer RAM nicht "
            "ermittelbar -- sicherheitshalber aktiviert.")
        return True

    ratio = file_size / available_ram
    decision = ratio > LOW_MEMORY_RAM_RATIO
    log(
        f"Low-memory-Modus (automatisch): Eingabedatei {file_size / (1024 ** 3):.2f} GB, "
        f"verfuegbarer System-RAM {available_ram / (1024 ** 3):.2f} GB "
        f"({ratio * 100:.0f}%) -- {'aktiviert' if decision else 'deaktiviert'} "
        f"(ctq empfiehlt --low-memory ab >{int(LOW_MEMORY_RAM_RATIO * 100)}% des verfuegbaren RAM; "
        "betrifft nur System-RAM, nicht VRAM -- die GPU-Verarbeitung streamt bei ctq ohnehin "
        "immer nur einen Tensor auf einmal)."
    )
    return decision


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def build_ctq_command(program, input_path, output_path, format_key, exclude_arg,
                      extra_builtin_flag, convrot_groupsize, block_size, extra_args, low_memory):
    """Builds the ctq argument list. Used by the real run AND by the
    command preview so that both are guaranteed to be identical."""
    fmt = FORMAT_PRESETS[format_key]
    cmd = [program, "-i", input_path, "-o", output_path]
    cmd += list(fmt["flags"])
    if fmt.get("needs_groupsize"):
        cmd += ["--convrot-group-size", str(convrot_groupsize)]
    if fmt.get("needs_blocksize"):
        cmd += ["--block_size", str(block_size)]
    cmd += ["--comfy_quant", "--save-quant-metadata"]
    if low_memory:
        cmd += ["--low-memory"]
    if exclude_arg:
        cmd += ["--exclude-layers", exclude_arg]
    if extra_builtin_flag:
        cmd += [extra_builtin_flag]
    if extra_args:
        cmd += extra_args.split()
    return cmd


def run_job(job_id, input_path, output_path, format_key, exclude_arg, extra_builtin_flag,
            convrot_groupsize, block_size, extra_args, low_memory_mode):
    started_at = datetime.datetime.now()
    log_filename = f"{started_at.strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}_{Path(output_path).stem}.log"
    log_path = LOG_DIR / log_filename
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        with JOBS_LOCK:
            JOBS[job_id]["log"].append(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def set_status(status, **kw):
        with JOBS_LOCK:
            JOBS[job_id]["status"] = status
            if status in FINAL_STATES:
                JOBS[job_id]["finished_at"] = time.time()
            JOBS[job_id].update(kw)

    with JOBS_LOCK:
        JOBS[job_id]["log_file"] = log_filename
    # Only a file that this job created itself may be deleted on cancel --
    # never a file the user already had.
    output_existed_before = os.path.exists(output_path)

    def cancel_requested():
        with JOBS_LOCK:
            return JOBS[job_id].get("cancel_requested", False)

    def finish_cancelled():
        log("--- Job vom Nutzer abgebrochen ---")
        if not output_existed_before and os.path.exists(output_path):
            try:
                os.remove(output_path)
                log(f"Unvollstaendige Ausgabedatei entfernt: {output_path}")
            except OSError as e:
                log(f"Unvollstaendige Ausgabedatei konnte nicht entfernt werden: {e}")
        set_status("cancelled", error="Job abgebrochen.")

    log(f"=== Quantisierungs-Job gestartet: {started_at.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(f"Eingabe:  {input_path}")
    log(f"Ausgabe:  {output_path}")
    log(f"Format:   {format_key}")

    acquired = RUN_LOCK.acquire(blocking=False)
    try:
        if not acquired:
            log("FEHLER: Es laeuft bereits ein anderer Quantisierungs-Job.")
            set_status("error", error="Es laeuft bereits ein anderer Quantisierungs-Job. Bitte warten, bis er fertig ist.")
            return

        if not CTQ_EXE:
            log("FEHLER: ctq wurde nicht gefunden.")
            set_status("error", error="ctq wurde nicht gefunden (Scripts-Ordner neben python.exe). "
                                       "Erst 'pip install convert-to-quant' ausfuehren.")
            return

        fmt = FORMAT_PRESETS[format_key]
        low_memory = resolve_low_memory_flag(low_memory_mode, input_path, log)
        cmd = build_ctq_command(CTQ_EXE, input_path, output_path, format_key, exclude_arg,
                                extra_builtin_flag, convrot_groupsize, block_size, extra_args,
                                low_memory)

        log("Befehl: " + " ".join(cmd))
        set_status("running", progress=0, started_at=time.time())

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        if cancel_requested():
            finish_cancelled()
            return
        # On Linux/macOS ctq gets its own process group so that "Cancel" also ends
        # any child processes (Windows uses taskkill /T for that).
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, start_new_session=(os.name != "nt"),
        )
        with JOBS_LOCK:
            JOBS[job_id]["proc"] = proc
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                log(line)
            m = re.search(r"\((\d+)/(\d+)\)", line)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                if total:
                    with JOBS_LOCK:
                        JOBS[job_id]["progress"] = int(cur / total * 100)
        proc.stdout.close()
        ret = proc.wait()
        with JOBS_LOCK:
            JOBS[job_id].pop("proc", None)

        if cancel_requested():
            finish_cancelled()
            return

        if ret != 0:
            log(f"FEHLER: ctq wurde mit Fehlercode {ret} beendet.")
            set_status("error", error=f"ctq wurde mit Fehlercode {ret} beendet -- siehe Log oben.")
            return

        if fmt.get("marker_fix"):
            log("--- Bereinige Quantisierungs-Marker ---")
            fix_markers(output_path, log)

        log("--- Fertig ---")
        set_status("done", progress=100, output_path=output_path)
    except Exception as e:
        log(f"FEHLER: Unerwarteter Fehler: {e}")
        set_status("error", error=f"Unerwarteter Fehler: {e}")
    finally:
        duration = (datetime.datetime.now() - started_at).total_seconds()
        with JOBS_LOCK:
            final_status = JOBS[job_id].get("status")
        log(f"=== Job beendet nach {duration:.1f}s, Status: {final_status} ===")
        log_file.close()
        if acquired:
            RUN_LOCK.release()


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


TEMPLATE_PATH = SCRIPT_DIR / "templates" / "index.html"
STATIC_DIR = SCRIPT_DIR / "static"

MISSING_FILES_HTML = """<!DOCTYPE html><meta charset="utf-8"><title>GFlava-Quant</title>
<body style="font-family:system-ui,sans-serif;max-width:640px;margin:12vh auto;padding:0 20px;line-height:1.5">
<h1>Dateien fehlen / Files missing</h1>
<p>GFlava-Quant besteht ab Version 1.3 aus mehreren Dateien. Neben <code>quant_server.py</code> muessen die
Ordner <code>templates</code> und <code>static</code> liegen (zusammen mit <code>fonts</code> und <code>icon.png</code>).
Bitte den kompletten Ordner kopieren.</p>
<p>Since version 1.3 GFlava-Quant consists of several files. Next to <code>quant_server.py</code> the folders
<code>templates</code> and <code>static</code> must be present (plus <code>fonts</code> and <code>icon.png</code>).
Please copy the complete folder.</p></body>"""


def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/")
def index():
    try:
        html = TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError:
        return Response(MISSING_FILES_HTML, status=500, mimetype="text/html")
    return _no_cache(Response(html.replace("__APP_VERSION__", APP_VERSION), mimetype="text/html"))


@app.route("/static/<path:filename>")
def static_files(filename):
    return _no_cache(send_from_directory(STATIC_DIR, filename))


@app.route("/icon.png")
def icon():
    return send_from_directory(SCRIPT_DIR, "icon.png", mimetype="image/png")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(SCRIPT_DIR, "icon.png", mimetype="image/png")


@app.route("/fonts/<path:filename>")
def fonts(filename):
    return send_from_directory(os.path.join(SCRIPT_DIR, "fonts"), filename, mimetype="font/woff2")


@app.route("/api/config")
def api_config():
    return jsonify({
        "ctq_found": CTQ_EXE is not None,
        "ctq_path": CTQ_EXE,
        "formats": {
            k: {
                "label": v["label"], "label_en": v.get("label_en", v["label"]),
                "description": v["description"], "description_en": v.get("description_en", v["description"]),
                "verified": v["verified"],
            }
            for k, v in FORMAT_PRESETS.items()
        },
        "excludes": {
            k: {
                "label": v["label"], "label_en": v.get("label_en", v["label"]),
                "kind": v["kind"], "value": v["value"],
                "chip": v.get("chip"), "verified": v.get("verified"),
                "filter_desc": v.get("filter_desc"),
                "filter_desc_note_de": v.get("filter_desc_note_de"),
                "filter_desc_note_en": v.get("filter_desc_note_en"),
            }
            for k, v in EXCLUDE_PRESETS.items()
        },
    })


@app.route("/api/ctq_help")
def api_ctq_help():
    """Shows the real help that is authoritative for the installed ctq version,
    instead of us having to guess in this tool which flags currently exist."""
    if not CTQ_EXE:
        return jsonify({"error": "ctq wurde nicht gefunden."}), 400
    try:
        result = subprocess.run([CTQ_EXE, "--help"], capture_output=True, text=True, timeout=20)
        text = (result.stdout or "") + (result.stderr or "")
        if not text.strip():
            text = "(ctq --help hat keine Ausgabe geliefert)"
        return jsonify({"help": text})
    except Exception as e:
        return jsonify({"error": f"Konnte 'ctq --help' nicht ausfuehren: {e}"}), 500


PLACEHOLDER_INPUT = "MODELL.safetensors"


def _clean_path(value):
    """Cleans up a typed path: strips whitespace and quotes, turns a leading ~ into the
    home folder (Linux/macOS users type ~/models/..., the shell doesn't do that here)."""
    p = (value or "").strip().strip('"')
    return os.path.expanduser(p) if p.startswith("~") else p


def parse_quantize_request(data, strict=True):
    """Reads and validates the form values -- shared by starting a job
    (strict=True) and the command preview (strict=False, where e.g. the input file
    may still be missing). Returns (params, None) or
    (None, (error_text, http_status))."""
    input_path = _clean_path(data.get("input_path"))
    output_path = _clean_path(data.get("output_path"))
    format_key = data.get("format", "int8_convrot")
    exclude_key = data.get("exclude_preset", "none")
    custom_regex = (data.get("custom_regex") or "").strip()
    convrot_groupsize = data.get("convrot_groupsize") or 256
    block_size = data.get("block_size") or 128
    extra_args = (data.get("extra_args") or "").strip()
    low_memory_mode = data.get("low_memory_mode") or "auto"

    if low_memory_mode not in ("auto", "on", "off"):
        return None, ("low_memory_mode muss 'auto', 'on' oder 'off' sein.", 400)

    input_exists = bool(input_path) and os.path.isfile(input_path)
    if strict:
        if not input_path or not input_exists:
            return None, (f"Eingabedatei nicht gefunden: {input_path}", 400)
        if not input_path.lower().endswith(".safetensors"):
            return None, ("Die Eingabedatei muss eine .safetensors-Datei sein.", 400)
    if format_key not in FORMAT_PRESETS:
        return None, ("Unbekanntes Quantisierungs-Format.", 400)

    try:
        convrot_groupsize = int(convrot_groupsize)
    except (TypeError, ValueError):
        return None, ("ConvRot-Gruppengroesse muss eine Zahl sein.", 400)
    # ctq strictly requires a power of 4 here (per `ctq --help-experimental`).
    if FORMAT_PRESETS[format_key].get("needs_groupsize") and convrot_groupsize not in (4, 16, 64, 256, 1024):
        return None, ("ConvRot-Gruppengroesse muss 4, 16, 64, 256 oder 1024 sein (Potenz von 4).", 400)

    try:
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return None, ("Block-Groesse muss eine positive ganze Zahl sein.", 400)

    cmd_input = input_path or PLACEHOLDER_INPUT
    if not output_path:
        base, ext = os.path.splitext(cmd_input)
        output_path = f"{base}_{format_key}{ext}"

    if strict:
        # Otherwise ctq would overwrite the not-yet-quantized original file --
        # irreversibly, since there is no backup.
        if os.path.normcase(os.path.abspath(output_path)) == os.path.normcase(os.path.abspath(input_path)):
            return None, ("Ausgabedatei darf nicht mit der Eingabedatei identisch sein -- "
                          "das wuerde dein unquantisiertes Original unwiderruflich ueberschreiben.", 400)
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.isdir(out_dir):
            return None, (f"Zielordner existiert nicht: {out_dir}", 400)

    exclude_arg = None
    extra_builtin_flag = None
    if exclude_key == "custom":
        exclude_arg = custom_regex or None
    else:
        preset = EXCLUDE_PRESETS.get(exclude_key)
        if preset:
            if preset["kind"] == "regex":
                exclude_arg = preset["value"]
            elif preset["kind"] == "builtin":
                extra_builtin_flag = preset["value"]

    return {
        "input_path": cmd_input, "input_exists": input_exists, "output_path": output_path,
        "format_key": format_key, "exclude_arg": exclude_arg, "extra_builtin_flag": extra_builtin_flag,
        "convrot_groupsize": convrot_groupsize, "block_size": block_size,
        "extra_args": extra_args, "low_memory_mode": low_memory_mode,
    }, None


def _norm_path(p):
    return os.path.normcase(os.path.abspath(p))


def _prune_finished_jobs():
    """Remove finished jobs from memory except for the last MAX_FINISHED_JOBS
    (their complete log is stored permanently under logs/ anyway).
    Call only while holding JOBS_LOCK."""
    finished = sorted((v.get("finished_at", 0), k) for k, v in JOBS.items() if v.get("status") in FINAL_STATES)
    for _, k in finished[:-MAX_FINISHED_JOBS]:
        del JOBS[k]


def _queue_worker_loop():
    while True:
        with QUEUE_COND:
            while not JOB_QUEUE:
                QUEUE_COND.wait()
            job_id = JOB_QUEUE.pop(0)
            job = JOBS.get(job_id)
            if not job or job.get("status") != "queued":
                continue
            job["status"] = "starting"
            params = job["params"]
        try:
            run_job(job_id, *params)
        except Exception as e:  # run_job catches almost everything itself -- this is only the safety net
            with JOBS_LOCK:
                JOBS[job_id].update({"status": "error", "error": f"Unerwarteter Fehler: {e}",
                                     "finished_at": time.time()})


def _ensure_queue_worker():
    with JOBS_LOCK:
        t = _QUEUE_WORKER["thread"]
        if t is not None and t.is_alive():
            return
        t = threading.Thread(target=_queue_worker_loop, name="quant-queue-worker", daemon=True)
        _QUEUE_WORKER["thread"] = t
        t.start()


@app.route("/api/quantize", methods=["POST"])
def api_quantize():
    """Queues a job. If nothing else is active, it starts immediately."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Ungueltige Anfrage."}), 400

    p, err = parse_quantize_request(data, strict=True)
    if err:
        return jsonify({"error": err[0]}), err[1]

    job_id = str(uuid.uuid4())
    job = {
        "status": "queued", "log": [], "progress": 0,
        "name": Path(p["input_path"]).name, "format": p["format_key"],
        "input_path": p["input_path"], "output_path": p["output_path"],
        "created_at": time.time(),
        "params": (p["input_path"], p["output_path"], p["format_key"], p["exclude_arg"],
                   p["extra_builtin_flag"], p["convrot_groupsize"], p["block_size"],
                   p["extra_args"], p["low_memory_mode"]),
    }
    with QUEUE_COND:
        out_norm = _norm_path(p["output_path"])
        for other in JOBS.values():
            if other.get("status") in ACTIVE_STATES and _norm_path(other.get("output_path", "")) == out_norm:
                return jsonify({"error": "Fuer diese Ausgabedatei gibt es schon einen Auftrag in der Warteschlange. "
                                         "Waehle einen anderen Ausgabenamen."}), 409
        _prune_finished_jobs()
        ahead = sum(1 for j in JOBS.values() if j.get("status") in ACTIVE_STATES)
        JOBS[job_id] = job
        JOB_QUEUE.append(job_id)
        QUEUE_COND.notify()
    _ensure_queue_worker()
    return jsonify({"job_id": job_id, "output_path": p["output_path"], "position": ahead + 1, "queued": ahead > 0})


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Shows the ctq command a job with these settings would use --
    built with the same function as the real run."""
    data = request.get_json(force=True, silent=True) or {}
    p, err = parse_quantize_request(data, strict=False)
    if err:
        return jsonify({"error": err[0]}), err[1]
    mode = p["low_memory_mode"]
    if mode == "on":
        low_memory = True
    elif mode == "off":
        low_memory = False
    elif p["input_exists"]:
        low_memory = resolve_low_memory_flag("auto", p["input_path"], lambda m: None)
    else:
        low_memory = False  # "automatic" cannot be decided without a file
    cmd = build_ctq_command("ctq", p["input_path"], p["output_path"], p["format_key"], p["exclude_arg"],
                            p["extra_builtin_flag"], p["convrot_groupsize"], p["block_size"],
                            p["extra_args"], low_memory)
    return jsonify({
        "program": "ctq", "args": cmd[1:], "input_missing": not p["input_exists"],
        "low_memory_pending": mode == "auto" and not p["input_exists"],
    })


def _job_summary(job_id, job, queue_pos):
    return {
        "id": job_id, "name": job.get("name"), "format": job.get("format"),
        "status": job.get("status"), "progress": job.get("progress", 0),
        "error": job.get("error"), "output_path": job.get("output_path"),
        "log_file": job.get("log_file"), "position": queue_pos.get(job_id),
        "created_at": job.get("created_at"), "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }


@app.route("/api/queue")
def api_queue():
    """All active and the most recently finished jobs (without log lines)."""
    with JOBS_LOCK:
        queue_pos = {jid: i + 1 for i, jid in enumerate(JOB_QUEUE)}
        items = [_job_summary(jid, j, queue_pos) for jid, j in JOBS.items()]
    running = [i for i in items if i["status"] in ("starting", "running")]
    queued = sorted((i for i in items if i["status"] == "queued"), key=lambda i: i["created_at"] or 0)
    done = sorted((i for i in items if i["status"] in FINAL_STATES), key=lambda i: i["finished_at"] or 0, reverse=True)
    return jsonify({"jobs": running + queued + done, "active": bool(running or queued)})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Unbekannte job_id."}), 404
        return jsonify({
            "status": job.get("status"),
            "progress": job.get("progress", 0),
            "log": job.get("log", [])[-300:],
            "output_path": job.get("output_path"),
            "error": job.get("error"),
            "log_file": job.get("log_file"),
            "name": job.get("name"),
            "format": job.get("format"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        })


@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Unbekannte job_id."}), 404
        status = job.get("status")
        if status == "queued":
            # Not started yet: there is neither a process nor an output file
            # to clean up -- just take it out of the queue.
            if job_id in JOB_QUEUE:
                JOB_QUEUE.remove(job_id)
            job.update({"status": "cancelled", "error": "Job abgebrochen.", "finished_at": time.time()})
            return jsonify({"ok": True})
        if status not in ("starting", "running"):
            return jsonify({"error": "Der Job laeuft nicht mehr."}), 409
        job["cancel_requested"] = True
        proc = job.get("proc")
    if proc is not None:
        _kill_process_tree(proc)
    return jsonify({"ok": True})


def _kill_process_tree(proc):
    """Ends ctq together with its child processes. Windows: ctq.exe is only a launcher for its
    own python.exe child process -- without /T the quantization would keep running on the GPU.
    Linux/macOS: ctq runs in its own process group (start_new_session), which is ended as a whole."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


def _terminate_running_jobs():
    """When the server exits (Ctrl+C, kill, terminal closed), take running ctq processes
    along. Because of the separate process group on Linux/macOS, the terminal's signal no
    longer reaches them on its own; ctq would otherwise keep computing orphaned on the GPU."""
    with JOBS_LOCK:
        procs = [j["proc"] for j in JOBS.values() if j.get("proc") is not None]
    for proc in procs:
        try:
            _kill_process_tree(proc)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Presets: custom combinations of format, layer protection and Advanced values,
# stored permanently in presets.json next to this script (config.json is left
# untouched by this). Input/output files are deliberately NOT part of them.
# ---------------------------------------------------------------------------

PRESETS_PATH = SCRIPT_DIR / "presets.json"
PRESETS_LOCK = threading.Lock()
PRESET_MAX = 50


def load_presets():
    try:
        data = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_presets(presets):
    tmp = PRESETS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, PRESETS_PATH)


def clean_preset_settings(s):
    """Only known fields, only valid values. Returns (settings, None) or (None, error)."""
    if not isinstance(s, dict):
        return None, "Ungueltige Einstellungen."
    fmt = s.get("format", "int8_convrot")
    if fmt not in FORMAT_PRESETS:
        return None, "Unbekanntes Quantisierungs-Format."
    ex = s.get("exclude_preset", "none")
    if ex not in EXCLUDE_PRESETS:
        return None, "Unbekannter Layer-Schutz."
    mem = s.get("low_memory_mode", "auto")
    if mem not in ("auto", "on", "off"):
        return None, "Ungueltiger Low-Memory-Modus."
    try:
        group = int(s.get("convrot_groupsize") or 256)
        block = int(s.get("block_size") or 128)
    except (TypeError, ValueError):
        return None, "Gruppen- und Blockgroesse muessen Zahlen sein."
    return {
        "format": fmt, "exclude_preset": ex, "low_memory_mode": mem,
        "convrot_groupsize": group, "block_size": block,
        "custom_regex": str(s.get("custom_regex") or "")[:2000],
        "extra_args": str(s.get("extra_args") or "")[:2000],
    }, None


@app.route("/api/presets")
def api_presets():
    with PRESETS_LOCK:
        presets = load_presets()
    return jsonify({"presets": [{"name": n, "settings": v} for n, v in sorted(presets.items(), key=lambda kv: kv[0].lower())]})


@app.route("/api/presets", methods=["POST"])
def api_presets_save():
    data = request.get_json(force=True, silent=True) or {}
    name = str(data.get("name") or "").strip()
    if not name or len(name) > 60:
        return jsonify({"error": "Der Name muss 1 bis 60 Zeichen lang sein."}), 400
    settings, err = clean_preset_settings(data.get("settings"))
    if err:
        return jsonify({"error": err}), 400
    with PRESETS_LOCK:
        presets = load_presets()
        if name not in presets and len(presets) >= PRESET_MAX:
            return jsonify({"error": f"Mehr als {PRESET_MAX} Voreinstellungen sind nicht moeglich."}), 400
        presets[name] = settings
        try:
            save_presets(presets)
        except OSError as e:
            return jsonify({"error": f"Konnte presets.json nicht speichern: {e}"}), 500
    return jsonify({"ok": True})


@app.route("/api/presets/delete", methods=["POST"])
def api_presets_delete():
    data = request.get_json(force=True, silent=True) or {}
    name = str(data.get("name") or "")
    with PRESETS_LOCK:
        presets = load_presets()
        if name not in presets:
            return jsonify({"error": "Voreinstellung nicht gefunden."}), 404
        del presets[name]
        try:
            save_presets(presets)
        except OSError as e:
            return jsonify({"error": f"Konnte presets.json nicht speichern: {e}"}), 500
    return jsonify({"ok": True})


@app.route("/api/models")
def api_models():
    return jsonify({"models": scan_models(), "roots": CONFIG.get("model_roots", [])})


@app.route("/api/detect_model")
def api_detect_model():
    """Reads the tensor header of the selected file and checks it against known
    architecture signatures (see MODEL_DETECTION_REGISTRY). Called by the frontend
    when a model is selected, not when listing all models -- so the total number of
    models in the folder doesn't matter for speed."""
    path = _clean_path(request.args.get("path", ""))
    if not path or not os.path.isfile(path):
        return jsonify({"error": "Datei nicht gefunden."}), 400
    try:
        header = read_safetensors_header(path)
    except Exception as e:
        return jsonify({"error": f"Konnte Datei nicht lesen: {e}"}), 400
    keys = [k for k in header.keys() if k != "__metadata__"]
    info = analyze_safetensors_header(header)
    try:
        info["size_bytes"] = os.path.getsize(path)
    except OSError:
        info["size_bytes"] = None
    return jsonify({"preset_key": detect_model_type(keys), **info})


# ---------------------------------------------------------------------------
# Settings: model folders, ComfyUI path, ctq override -- everything configurable
# in the browser instead of hardcoded, stored permanently in config.json.
# ---------------------------------------------------------------------------


@app.route("/api/settings")
def api_settings():
    return jsonify({
        "model_roots": CONFIG.get("model_roots", []),
        "comfyui_root": CONFIG.get("comfyui_root", ""),
        "ctq_path_override": CONFIG.get("ctq_path_override"),
        "ctq_auto_path": find_ctq_executable(None),
        "ctq_found": CTQ_EXE is not None,
        "first_run": not CONFIG_EXISTED_AT_START,
    })


@app.route("/api/settings", methods=["POST"])
def api_settings_save():
    global CTQ_EXE
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Ungueltige Anfrage."}), 400

    # Validate EVERYTHING first without touching CONFIG -- otherwise, with e.g.
    # valid model_roots + an invalid comfyui_root in the same request, the
    # model_roots change would already be applied in the live server but never
    # saved to config.json because of the error (an inconsistent state that only
    # disappears at the next restart).
    new_model_roots = None
    if "model_roots" in data:
        roots = data["model_roots"]
        if not isinstance(roots, list):
            return jsonify({"error": "model_roots muss eine Liste sein."}), 400
        clean = []
        seen_keys = set()
        for i, r in enumerate(roots):
            if not isinstance(r, dict):
                continue
            path = _clean_path(r.get("path"))
            if not path:
                continue
            if not os.path.isdir(path):
                return jsonify({"error": f"Ordner existiert nicht: {path}"}), 400
            label = (r.get("label") or "").strip() or path
            key = re.sub(r"[^a-zA-Z0-9_]+", "_", (r.get("key") or "").strip() or f"root{i}").strip("_") or f"root{i}"
            base_key, n = key, 1
            while key in seen_keys:
                key = f"{base_key}_{n}"
                n += 1
            seen_keys.add(key)
            clean.append({"key": key, "label": label, "path": path})
        new_model_roots = clean

    new_comfyui_root = None
    if "comfyui_root" in data:
        new_comfyui_root = _clean_path(data["comfyui_root"])
        if new_comfyui_root and not os.path.isdir(new_comfyui_root):
            return jsonify({"error": f"ComfyUI-Ordner existiert nicht: {new_comfyui_root}"}), 400

    new_ctq_override = None
    if "ctq_path_override" in data:
        override = _clean_path(data["ctq_path_override"])
        if override and not os.path.isfile(override):
            return jsonify({"error": f"Datei existiert nicht: {override}"}), 400
        new_ctq_override = override or None

    # All validations passed -- only now apply and save.
    if "model_roots" in data:
        CONFIG["model_roots"] = new_model_roots
    if "comfyui_root" in data:
        CONFIG["comfyui_root"] = new_comfyui_root
    if "ctq_path_override" in data:
        CONFIG["ctq_path_override"] = new_ctq_override
        CTQ_EXE = find_ctq_executable(new_ctq_override)

    save_config()
    _invalidate_system_check_cache()  # the ComfyUI/ctq path may have changed
    return jsonify({"ok": True, "model_count": len(scan_models())})


@app.route("/api/browse_native", methods=["POST"])
def api_browse_native():
    """Opens a REAL Windows picker dialog via an embedded PowerShell, just like
    start_server.bat already does for the python.exe selection. An <input type=file>
    cannot return a real file system path for security reasons, hence the detour via
    the server (which only listens on 127.0.0.1). Blocks the request until the user
    picks/cancels -- threaded=True on app.run() makes sure that the rest of the
    server (e.g. a running quantization job) is unaffected.

    For folders a FolderBrowserDialog is deliberately NOT used -- under PowerShell/WinForms
    it shows the old "Browse for folder" tree dialog, not the modern Explorer style that
    you know from a file dialog (see the ctq.exe/python.exe selection). Instead the usual
    trick: a completely normal OpenFileDialog in the modern style, with files hidden
    entirely (the filter matches nothing), where you simply open the desired folder while
    a placeholder text stays in the file name field -- the folder path is extracted from
    that afterwards.
    """
    if os.name != "nt":
        # "unsupported": the UI then asks for the path via a text prompt.
        return jsonify({"error": "Nativer Dialog ist nur unter Windows verfuegbar.", "unsupported": True}), 400
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Ungueltige Anfrage."}), 400

    mode = data.get("mode", "folder")  # "folder" or "file"
    title = str(data.get("title") or "").replace("'", "''")
    initial_dir = str(data.get("initial_dir") or "").strip()
    initial_dir = initial_dir if os.path.isdir(initial_dir) else ""
    initial_dir_ps = initial_dir.replace("'", "''")
    dialog_filter = str(data.get("filter") or "All files|*.*").replace("'", "''")

    if mode == "folder":
        select_hint = str(data.get("select_hint") or "Select this folder").replace("'", "''")
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$f = New-Object System.Windows.Forms.OpenFileDialog; "
            f"$f.Title = '{title}'; "
            "$f.CheckFileExists = $false; $f.CheckPathExists = $true; "
            "$f.ValidateNames = $false; $f.AddExtension = $false; $f.Multiselect = $false; "
            "$f.Filter = \"Folders|`n\"; "
            f"$f.FileName = '{select_hint}'; "
            + (f"$f.InitialDirectory = '{initial_dir_ps}'; " if initial_dir_ps else "")
            + "if ($f.ShowDialog() -eq 'OK') { Write-Output ([System.IO.Path]::GetDirectoryName($f.FileName)) }"
        )
    elif mode == "file":
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$f = New-Object System.Windows.Forms.OpenFileDialog; "
            f"$f.Title = '{title}'; $f.Filter = '{dialog_filter}'; "
            + (f"$f.InitialDirectory = '{initial_dir_ps}'; " if initial_dir_ps else "")
            + "if ($f.ShowDialog() -eq 'OK') { Write-Output $f.FileName }"
        )
    else:
        return jsonify({"error": "mode muss 'folder' oder 'file' sein."}), 400

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Zeitueberschreitung beim Warten auf die Dialog-Auswahl."}), 504
    except Exception as e:
        return jsonify({"error": f"Dialog konnte nicht geoeffnet werden: {e}"}), 500

    if result.returncode != 0:
        return jsonify({"error": (result.stderr or "PowerShell-Dialog fehlgeschlagen.")[-1000:]}), 500

    picked = (result.stdout or "").strip()
    return jsonify({"path": picked or None})


# ---------------------------------------------------------------------------
# System check: checks whether ctq and the ComfyUI-INT8-Fast custom node
# are installed and whether updates are available for them.
# ---------------------------------------------------------------------------


def _run_cmd(cmd, cwd=None, timeout=20):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return -1, str(e)


def check_ctq_status():
    base = {"id": "ctq", "label": "convert_to_quant (ctq)", "label_en": "convert_to_quant (ctq)"}
    if not CTQ_EXE:
        return {**base, "status": "missing", "can_install": True, "can_update": False,
                "detail": "ctq wurde nicht gefunden -- ueber \"Installieren\" per pip nachinstallieren, "
                          "oder manuell: pip install convert_to_quant",
                "detail_en": "ctq was not found -- install it via pip using \"Install\" below, "
                             "or manually: pip install convert_to_quant"}

    code, out = _run_cmd([sys.executable, "-m", "pip", "index", "versions", "convert_to_quant"], timeout=25)
    installed_m = re.search(r"INSTALLED:\s*(\S+)", out)
    latest_m = re.search(r"LATEST:\s*(\S+)", out)
    if code != 0 or not installed_m:
        return {**base, "status": "ok", "can_install": False, "can_update": False,
                "detail": f"Gefunden unter {CTQ_EXE}. Versions-/Update-Pruefung nicht moeglich "
                          "(kein Internet oder pip-Fehler).",
                "detail_en": f"Found at {CTQ_EXE}. Could not check version/updates "
                             "(no internet or pip error)."}

    installed = installed_m.group(1)
    latest = latest_m.group(1) if latest_m else installed
    if installed == latest:
        return {**base, "status": "ok", "can_install": False, "can_update": False,
                "detail": f"Version {installed} installiert (aktuell). {CTQ_EXE}",
                "detail_en": f"Version {installed} installed (up to date). {CTQ_EXE}"}
    return {**base, "status": "warn", "can_install": False, "can_update": True,
            "update_available": True, "latest_version": latest,
            "detail": f"Version {installed} installiert, {latest} verfuegbar.",
            "detail_en": f"Version {installed} installed, {latest} available."}


def _int8fast_dir():
    root = (CONFIG.get("comfyui_root") or "").strip()
    if not root:
        return None
    return Path(root) / "custom_nodes" / INT8FAST_DIR_NAME


def check_int8fast_status():
    base = {"id": "int8fast", "label": "ComfyUI-INT8-Fast", "label_en": "ComfyUI-INT8-Fast"}
    d = _int8fast_dir()
    if d is None:
        return {**base, "status": "unknown", "can_install": False, "can_update": False,
                "detail": "ComfyUI-Ordner ist nicht konfiguriert -- unter Einstellungen eintragen.",
                "detail_en": "ComfyUI folder is not configured -- set it under Settings."}
    if not d.is_dir():
        return {**base, "status": "missing", "can_install": True, "can_update": False,
                "detail": f"Nicht installiert (erwartet unter {d}).",
                "detail_en": f"Not installed (expected at {d})."}
    if not (d / ".git").is_dir():
        return {**base, "status": "unknown", "can_install": False, "can_update": False,
                "detail": f"Ordner vorhanden ({d}), aber kein Git-Repo -- Update-Pruefung nicht moeglich.",
                "detail_en": f"Folder present ({d}), but not a git repo -- cannot check for updates."}

    code1, local = _run_cmd(["git", "rev-parse", "HEAD"], cwd=str(d), timeout=15)
    code2, remote = _run_cmd(["git", "ls-remote", "origin", "HEAD"], cwd=str(d), timeout=20)
    if code1 != 0 or code2 != 0:
        return {**base, "status": "ok", "can_install": False, "can_update": False,
                "detail": f"Installiert unter {d}. Update-Pruefung fehlgeschlagen (kein Internet?).",
                "detail_en": f"Installed at {d}. Update check failed (no internet?)."}

    local_hash = local.strip().split()[0] if local.strip() else ""
    remote_hash = remote.strip().split()[0] if remote.strip() else ""
    if remote_hash and local_hash and remote_hash != local_hash:
        return {**base, "status": "warn", "can_install": False, "can_update": True,
                "update_available": True, "latest_version": remote_hash[:7],
                "detail": f"Installiert unter {d}. Update verfuegbar.",
                "detail_en": f"Installed at {d}. Update available."}
    return {**base, "status": "ok", "can_install": False, "can_update": False,
            "detail": f"Installiert unter {d} (aktuell).",
            "detail_en": f"Installed at {d} (up to date)."}


def _version_tuple(v):
    """'v1.2.0' / '1.3' -> (1, 2, 0) / (1, 3, 0); None if there is no version number in it."""
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(v or ""))
    if not m:
        return None
    parts = [int(x) for x in m.group(1).split(".")][:3]
    return tuple(parts + [0] * (3 - len(parts)))


def check_app_update():
    """Compares APP_VERSION with the latest release on GitHub. Nothing is installed:
    the user only gets the link to the release."""
    base = {"id": "app", "label": "GFlava-Quant", "label_en": "GFlava-Quant",
            "can_install": False, "can_update": False, "release_url": GFLAVA_RELEASES_URL}
    req = urllib.request.Request(
        f"https://api.github.com/repos/{GFLAVA_REPO}/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": f"GFlava-Quant/{APP_VERSION}"},
    )
    try:
        # Python from python.org on macOS ships without certificates (HTTPS verification
        # then fails). certifi is present in every ComfyUI environment, so prefer it.
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            context = None
        with urllib.request.urlopen(req, timeout=10, context=context) as resp:
            release = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {**base, "status": "unknown",
                "detail": f"Version {APP_VERSION}. Update-Pruefung nicht moeglich "
                          "(kein Internet oder GitHub nicht erreichbar).",
                "detail_en": f"Version {APP_VERSION}. Could not check for updates "
                             "(no internet or GitHub unreachable)."}

    tag = str(release.get("tag_name") or "")
    latest, current = _version_tuple(tag), _version_tuple(APP_VERSION)
    url = release.get("html_url") or GFLAVA_RELEASES_URL
    if not str(url).startswith("https://github.com/"):
        url = GFLAVA_RELEASES_URL
    if latest is None or current is None:
        return {**base, "status": "unknown",
                "detail": f"Version {APP_VERSION}. Das neueste Release ({tag or '?'}) hat keine lesbare Versionsnummer.",
                "detail_en": f"Version {APP_VERSION}. The latest release ({tag or '?'}) has no readable version number."}
    latest_str = tag.lstrip("vV")
    if latest > current:
        return {**base, "status": "warn", "update_available": True, "latest_version": latest_str,
                "release_url": url,
                "detail": f"Version {APP_VERSION} installiert, {latest_str} verfuegbar.",
                "detail_en": f"Version {APP_VERSION} installed, {latest_str} available."}
    return {**base, "status": "ok", "release_url": url,
            "detail": f"Version {APP_VERSION} installiert (aktuell).",
            "detail_en": f"Version {APP_VERSION} installed (up to date)."}


# Result of the last check. The automatic check when the page loads uses it as long
# as it is at most an hour old -- otherwise every reload would query GitHub, PyPI and
# git (GitHub allows only 60 requests per hour without authentication).
SYSTEM_CHECK_MAX_AGE = 3600
SYSTEM_CHECK_CACHE = {"checks": None, "checked_at": 0.0}
SYSTEM_CHECK_LOCK = threading.Lock()


def _invalidate_system_check_cache():
    with SYSTEM_CHECK_LOCK:
        SYSTEM_CHECK_CACHE["checks"] = None


@app.route("/api/system_check")
def api_system_check():
    """?cached=1: automatic check, may reuse a fresh result.
    Without it: the button in Settings, always checks again."""
    use_cache = request.args.get("cached") == "1"
    with SYSTEM_CHECK_LOCK:
        fresh = (SYSTEM_CHECK_CACHE["checks"] is not None
                 and time.time() - SYSTEM_CHECK_CACHE["checked_at"] < SYSTEM_CHECK_MAX_AGE)
        if not (use_cache and fresh):
            SYSTEM_CHECK_CACHE["checks"] = [check_app_update(), check_ctq_status(), check_int8fast_status()]
            SYSTEM_CHECK_CACHE["checked_at"] = time.time()
        return jsonify({"checks": SYSTEM_CHECK_CACHE["checks"],
                        "checked_at": SYSTEM_CHECK_CACHE["checked_at"],
                        "app_version": APP_VERSION})


@app.route("/api/system_check/action", methods=["POST"])
def api_system_check_action():
    global CTQ_EXE
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Ungueltige Anfrage."}), 400
    check_id = data.get("id")
    action = data.get("action")
    _invalidate_system_check_cache()  # after install/update the old result is no longer correct

    if check_id == "int8fast" and action == "install":
        d = _int8fast_dir()
        if d is None:
            return jsonify({"error": "ComfyUI-Ordner ist nicht konfiguriert."}), 400
        if d.is_dir():
            return jsonify({"error": "Ist bereits installiert."}), 400
        d.parent.mkdir(parents=True, exist_ok=True)
        code, out = _run_cmd(["git", "clone", INT8FAST_REPO_URL, str(d)], timeout=180)
        if code != 0:
            return jsonify({"error": f"git clone fehlgeschlagen: {out[-2000:]}"}), 500
        return jsonify({"ok": True, "output": out[-2000:]})

    if check_id == "int8fast" and action == "update":
        d = _int8fast_dir()
        if d is None or not (d / ".git").is_dir():
            return jsonify({"error": "Nicht installiert oder kein Git-Repo."}), 400
        code, out = _run_cmd(["git", "pull"], cwd=str(d), timeout=60)
        if code != 0:
            return jsonify({"error": f"git pull fehlgeschlagen: {out[-2000:]}"}), 500
        return jsonify({"ok": True, "output": out[-2000:]})

    if check_id == "ctq" and action in ("install", "update"):
        code, out = _run_cmd([sys.executable, "-m", "pip", "install", "-U", "convert_to_quant"], timeout=180)
        if code != 0:
            return jsonify({"error": f"pip install fehlgeschlagen: {out[-2000:]}"}), 500
        CTQ_EXE = find_ctq_executable(CONFIG.get("ctq_path_override"))
        if not CTQ_EXE:
            return jsonify({"error": "pip install lief durch, aber ctq.exe wurde danach trotzdem nicht "
                                      "gefunden -- evtl. Server neu starten oder Pfad manuell eintragen."}), 500
        return jsonify({"ok": True, "output": out[-2000:]})

    return jsonify({"error": "Unbekannte Aktion."}), 400


@app.route("/api/logs")
def api_logs():
    try:
        files = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]
    except OSError:
        files = []
    logs = []
    for p in files:
        try:
            stat = p.stat()
        except OSError:
            continue  # e.g. deleted between listing and here
        logs.append({"name": p.name, "size": stat.st_size, "mtime": stat.st_mtime})
    return jsonify({"logs": logs})


@app.route("/api/logs/<path:name>")
def api_log_file(name):
    # Only the file name counts -- prevents access outside of logs/.
    fname = Path(name).name
    p = LOG_DIR / fname
    if not p.is_file():
        return jsonify({"error": "Log nicht gefunden."}), 404
    return Response(p.read_text(encoding="utf-8", errors="replace"), mimetype="text/plain; charset=utf-8")


if __name__ == "__main__":
    print("ctq gefunden unter:", CTQ_EXE or "NICHT GEFUNDEN")
    if not TEMPLATE_PATH.is_file() or not (STATIC_DIR / "app.js").is_file():
        print("WARNUNG: templates/index.html oder static/app.js fehlt -- bitte den kompletten Ordner kopieren.")
    print("Oeffne im Browser: http://127.0.0.1:8877")
    atexit.register(_terminate_running_jobs)
    for _sig in ("SIGTERM", "SIGHUP"):  # SIGHUP only exists on Linux/macOS
        if hasattr(signal, _sig):
            signal.signal(getattr(signal, _sig), lambda signum, frame: sys.exit(0))
    # threaded=True so that an open native folder/file dialog (blocks the request
    # until the user picks/cancels) doesn't freeze the whole server -- a quantization
    # job running in parallel should still be able to deliver status updates.
    app.run(host="127.0.0.1", port=8877, debug=False, threaded=True)
