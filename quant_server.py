"""
Standalone lokales Quantisierungs-Tool fuer ComfyUI-Modelle.

Startet einen kleinen lokalen Webserver (nur auf deinem eigenen PC erreichbar,
127.0.0.1) mit einer Oberflaeche im Browser. Du gibst dort den Pfad zu einer
bf16-Modelldatei an, waehlst das gewuenschte Quantisierungsformat, und das
Skript ruft im Hintergrund `ctq` (convert_to_quant) mit den passenden
Parametern auf.

Fuer INT8-Tensorwise/ConvRot wird danach automatisch der Marker-Fix
angewendet, den wir gemeinsam herausgefunden haben (ctq schreibt zwei
zusaetzliche Felder in den `.comfy_quant`-Marker, die ComfyUI nicht kennt
und deshalb den ganzen Marker verwirft -> "unet unexpected"-Warnung).
Dieser Schritt kuerzt den Marker auf genau die Felder, die die offizielle
Comfy-Org-Datei auch benutzt.

WICHTIG: Das hier ist eine lokale Anwendung, kein oeffentliches Web-Tool.
Sie liest/schreibt Dateien auf DEINEM Rechner und ruft ein Programm
(ctq) auf DEINEM Rechner auf. Das kann und darf nicht als gehostete
Webseite laufen -- nur lokal, mit dem Python, das auch dein ComfyUI benutzt.

Start (im ComfyUI-Python):
    "H:\\ComfyUI_windows_portable\\ComfyUI-Easy-Install\\python_embeded\\python.exe" -m pip install flask
    "H:\\ComfyUI_windows_portable\\ComfyUI-Easy-Install\\python_embeded\\python.exe" quant_server.py

Danach im Browser oeffnen: http://127.0.0.1:8877
"""

import datetime
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

# static_folder=None: /static/ wird unten von einer eigenen Route bedient (ohne Browser-Cache),
# damit Aenderungen an style.css/app.js nach einem Neuladen sofort sichtbar sind.
app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# Konfiguration: Modell-Ordner, ComfyUI-Installation, ctq-Pfad-Override.
# Wird dauerhaft in config.json neben diesem Skript gespeichert, damit nichts
# hart im Code hinterlegt werden muss und Aenderungen einen Server-Neustart
# ueberleben. Beim allerersten Start gibt es noch keine Modell-Ordner -- die
# Einstellungen oeffnen sich dann automatisch mit einem Hinweis.
# ---------------------------------------------------------------------------

APP_VERSION = "1.3"
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CONFIG_PATH = SCRIPT_DIR / "config.json"

DEFAULT_MODEL_ROOTS = []


def _guess_comfyui_root():
    """Raet die ComfyUI-Installation ueber den Python-Interpreter (typisches
    ComfyUI-Portable-Layout: python_embeded und ComfyUI liegen nebeneinander).
    Nur ein Vorschlag -- der Nutzer kann das in den Einstellungen aendern."""
    candidate = Path(sys.executable).parent.parent / "ComfyUI"
    return str(candidate) if candidate.is_dir() else ""


def load_config():
    """Liefert (config, existed_before) -- existed_before sagt, ob config.json
    schon vor diesem Start da war (fuer die einmalige Erstlauf-Anzeige)."""
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
# ctq finden
# ---------------------------------------------------------------------------


def find_ctq_executable(override=None):
    """Sucht ctq(.exe). Ein manueller Override (aus den Einstellungen) geht
    vor; sonst wird im Scripts-Ordner neben dem aktuellen Python-Interpreter
    gesucht -- das passt zu jeder venv/embedded-python-Installation, egal wo
    genau ComfyUI liegt, wir muessen den Pfad nicht hart hinterlegen.
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
        if c.exists():
            return str(c)
    return None


CTQ_EXE = find_ctq_executable(CONFIG.get("ctq_path_override"))
INT8FAST_REPO_URL = "https://github.com/BobJohnson24/ComfyUI-INT8-Fast"
INT8FAST_DIR_NAME = "ComfyUI-INT8-Fast"
# Eigenes Repo: dort wird bei der Update-Pruefung das neueste Release abgefragt.
GFLAVA_REPO = "Freeman79680/GFlava-Quant"
GFLAVA_RELEASES_URL = f"https://github.com/{GFLAVA_REPO}/releases"

# ---------------------------------------------------------------------------
# Quantisierungs-Formate
#
# Jeder Eintrag: die ctq-Flags, die wir tatsaechlich in der Doku/den Beispielen
# von convert_to_quant bestaetigt gesehen haben. "verified" markiert, was wir
# in unserem eigenen Test tatsaechlich Ende-zu-Ende in ComfyUI geladen und
# geprueft haben (aktuell nur INT8 ConvRot). Die anderen Formate sind nach
# ctq's eigener Doku korrekt aufgerufen, aber wir haben ihr Marker-Format
# nicht selbst verifiziert -- deshalb bleibt bei denen der Marker-Fix aus.
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
        # --scaling_mode block ist Pflicht, sonst ignoriert ctq --block_size komplett und
        # quantisiert tensor-weise (per `ctq --help-experimental` bestaetigt).
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

# Layer-Ausschluss-Presets. "regex" = eigener Wert fuer --exclude-layers
# (bei "verified": true von uns Ende-zu-Ende getestet, sonst ein aus
# bekannter/oeffentlich dokumentierter Architektur abgeleitetes Muster, das
# wir NICHT selbst gegen eine offizielle Datei verglichen haben). "builtin" =
# einer von ctq's eigenen Modell-Flags (siehe `ctq -hf`); die werden als
# eigenstaendiges Flag angehaengt, nicht als --exclude-layers-Wert. "chip" =
# Kurzname fuer die Schnellauswahl-Kacheln oben im Formular (optional).
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
    # -- ctq eigene Modell-Presets (aus `ctq --help-filters`, Stand unserer Installation). --
    # "filter_desc" ist woertlich die Beschreibung, die ctq selbst fuer dieses Preset ausgibt --
    # nicht von uns interpretiert, damit hier nichts geraten wird. Ist bereits Englisch (ctq's
    # eigene CLI-Ausgabe) und wird daher in beiden Sprachen unveraendert angezeigt.
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

# Welche Felder wir im .comfy_quant-Marker behalten, abhaengig vom "format"-
# Wert, der IM Marker selbst steht. Nur fuer Formate, die wir selbst gegen
# eine offizielle Comfy-Org-Datei verglichen haben.
MARKER_ALLOWED_FIELDS = {
    "int8_tensorwise": ["format", "convrot", "convrot_groupsize"],
}

# ---------------------------------------------------------------------------
# Logs: jeder Quantisierungs-Lauf wird komplett und dauerhaft unter logs/
# gespeichert (bleibt auch nach einem Server-Neustart erhalten), damit man
# einen Lauf im Nachhinein noch nachvollziehen kann.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Modell-Ordner fuer das Auswahl-Dropdown. Kommen jetzt aus CONFIG (siehe
# Einstellungen im Browser), nicht mehr hart aus dem Code. Werden rekursiv
# nach .safetensors durchsucht; das Dropdown zeigt zu jedem Modell an, aus
# welchem Ordner (und ggf. welchem Unterordner) es stammt.
# ---------------------------------------------------------------------------


def scan_models():
    entries = []
    for root in CONFIG.get("model_roots", []):
        base = Path(root["path"])
        if not base.is_dir():
            continue
        try:
            found = list(base.rglob("*.safetensors"))
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
# Modelltyp-Erkennung: liest nur den JSON-Header einer .safetensors-Datei
# (kein Laden von Tensor-Daten, daher auch bei grossen Modellen sehr
# schnell) und prueft die Tensor-Namen gegen Signaturen, die entweder direkt
# aus ctq's eigenen, installierten MODEL_FILTERS-Konstanten stammen (echte
# Ground Truth -- siehe convert_to_quant/constants.py) oder von uns gegen
# eine echte Datei verifiziert wurden (Qwen-Image 2.1 Single-Stream: keine
# offizielle ctq-Konstante, da ctq's eigenes --qwen die AELTERE Dual-Stream-
# Variante meint). Jede Regel verlangt ALLE "require"-Substrings und KEINEN
# der "forbid"-Substrings in den Tensor-Namen -- lieber keine Erkennung als
# eine falsche. Architekturen ohne eine wirklich unterscheidbare Signatur
# (aktuell nur LENS -- zu aehnlich zu Qwen-Dual-Stream, teilt "img_mod"/
# "time_text_embed"/"img_in" ohne ein eigenes eindeutiges Merkmal) sind
# bewusst NICHT aufgenommen.
# ---------------------------------------------------------------------------


def _get_ctq_constants():
    try:
        from convert_to_quant import constants as ctq_constants
        return ctq_constants
    except Exception:
        return None


def build_model_detection_registry():
    registry = [
        # Gegen eine echte qwen_image_2.1_bf16.safetensors verifiziert: 32
        # Bloecke mit transformer_blocks.N.img_mlp.*, ein einzelnes
        # modulation.1.weight, txt_in.text_norm.weight -- kein img_mod/
        # txt_mod wie bei ctq's aelterer Dual-Stream-Variante (--qwen).
        {"preset_key": "qwen21_custom", "require": ["img_mlp", "txt_in.text_norm"], "forbid": ["img_mod", "txt_mod"]},
        # Gegen eine echte flux1-dev-bnb-nf4.safetensors verifiziert:
        # double_blocks/single_blocks mit img_mod.lin/txt_mod.lin pro Block --
        # anders als Flux.2 (double_stream_modulation_img/txt, kein img_mod)
        # und anders als ctq's Qwen-Dual-Stream (transformer_blocks statt
        # double_blocks/single_blocks).
        {"preset_key": "flux1_custom", "require": ["double_blocks", "single_blocks", "img_mod", "txt_mod"],
         "forbid": ["stream_modulation"]},
        # Gegen eine echte SDXL/Illustrious-Datei verifiziert: U-Net-Aufbau
        # (input_blocks/middle_block/output_blocks/label_emb) kommt bei keiner
        # DiT-Architektur (Flux/Qwen/Anima/Z-Image/...) vor -- voellig anderer
        # Aufbau, daher besonders sicher zu erkennen.
        {"preset_key": "sdxl_custom", "require": ["input_blocks", "middle_block", "output_blocks", "label_emb"],
         "forbid": []},
    ]

    if _get_ctq_constants() is not None:
        # Ab hier: Substrings 1:1 aus ctq's eigenen MODEL_FILTERS/*_LAYER_KEYNAMES
        # (siehe constants.py der installierten convert_to_quant-Version) --
        # nicht von uns geraten, sondern das, was ctq selbst pro Architektur
        # als charakteristisch ansieht. Gegen echte Anima-3.8B- und
        # Flux.2-Klein-Dateien zusaetzlich manuell verifiziert.
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
    """Liest nur den JSON-Header einer .safetensors-Datei (8-Byte-Laenge +
    Header), keine Tensor-Daten -- schnell auch bei sehr grossen Modellen."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        if header_size <= 0 or header_size > 200_000_000:
            raise ValueError("Ungueltiger oder zu grosser Header.")
        return json.loads(f.read(header_size))


def read_safetensors_keys(path):
    return [k for k in read_safetensors_header(path).keys() if k != "__metadata__"]


def analyze_safetensors_header(header):
    """Fuer die Groessenschaetzung in der Oberflaeche: wie viele Bytes liegen
    in 16-Bit- bzw. 32-Bit-Gewichten mit mindestens 2 Dimensionen (das sind
    die Tensoren, die ctq ueberhaupt quantisieren kann). Reine Naeherung."""
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

# Warteschlange: Auftraege laufen strikt nacheinander (GPU/RAM-Konflikte
# vermeiden), aber man kann beliebig viele einreihen. Ein einziger Worker-
# Thread holt sich den naechsten Auftrag, sobald der vorige beendet ist. Die
# Schlange liegt im Server, nicht im Browser -- ein geschlossener oder neu
# geladener Tab verliert also nichts.
JOB_QUEUE = []                      # job_ids in Reihenfolge, geschuetzt durch JOBS_LOCK
QUEUE_COND = threading.Condition(JOBS_LOCK)
ACTIVE_STATES = ("queued", "starting", "running")
FINAL_STATES = ("done", "error", "cancelled")
MAX_FINISHED_JOBS = 30              # so viele beendete Jobs bleiben fuer die Anzeige im Speicher
_QUEUE_WORKER = {"thread": None}


# ---------------------------------------------------------------------------
# Marker-Fix (Schritt 16 aus unserem manuellen Vorgehen, jetzt automatisiert)
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
# Low-Memory-Modus: ctq's --low-memory betrifft laut `ctq --help` ausdruecklich
# den System-RAM ("streaming tensor loading to reduce RAM usage, recommended
# for models >50% of available RAM"), NICHT den VRAM der Grafikkarte. Wir
# haben das im installierten ctq-Quellcode nachgeprueft
# (utils/memory_efficient_loader.py): ohne dieses Flag laedt ctq alle Tensoren
# vorab in einen Python-Dict im RAM, mit Flag liest es sie einzeln von der
# Platte. Die eigentliche GPU-Verarbeitung (converters/*.py) verschiebt so
# oder so immer nur einen Gewichts-Tensor auf einmal auf die GPU und ruft
# danach torch.cuda.empty_cache() auf -- das passiert automatisch und
# unabhaengig von --low-memory, es gibt also nichts VRAM-Bezogenes zu
# erkennen. Was wir automatisch erkennen koennen (und was ctq's eigener
# Empfehlung entspricht): ob die Eingabedatei mehr als die Haelfte des
# verfuegbaren System-RAM belegt.
# ---------------------------------------------------------------------------

LOW_MEMORY_RAM_RATIO = 0.5


def get_available_ram_bytes():
    """Verfuegbarer physischer System-RAM in Bytes, oder None wenn nicht
    bestimmbar (z.B. auf Nicht-Windows-Systemen)."""
    if os.name != "nt":
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
    """Entscheidet, ob --low-memory an ctq uebergeben wird. mode ist "auto"
    (RAM-basierte Erkennung), "on" (immer) oder "off" (nie)."""
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
# Job-Ausfuehrung
# ---------------------------------------------------------------------------


def build_ctq_command(program, input_path, output_path, format_key, exclude_arg,
                      extra_builtin_flag, convrot_groupsize, block_size, extra_args, low_memory):
    """Baut die ctq-Argumentliste. Wird vom echten Lauf UND von der
    Befehlsvorschau benutzt, damit beide garantiert identisch sind."""
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
    # Nur eine Datei, die dieser Job selbst angelegt hat, darf beim Abbruch
    # geloescht werden -- eine schon vorhandene Datei des Nutzers nie.
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
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
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
# Flask-Routen
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
    """Zeigt die echte, fuer die installierte ctq-Version verbindliche Hilfe an,
    statt dass wir hier im Tool raten muessen, welche Flags gerade existieren."""
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


def parse_quantize_request(data, strict=True):
    """Liest und prueft die Formularwerte -- gemeinsam fuer den Start eines
    Auftrags (strict=True) und die Befehlsvorschau (strict=False, dort darf
    z.B. die Eingabedatei noch fehlen). Liefert (params, None) oder
    (None, (fehlertext, http_status))."""
    input_path = (data.get("input_path") or "").strip().strip('"')
    output_path = (data.get("output_path") or "").strip().strip('"')
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
    # ctq verlangt hier zwingend eine Potenz von 4 (per `ctq --help-experimental`).
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
        # Sonst wuerde ctq die noch unquantisierte Originaldatei ueberschreiben --
        # unwiderruflich, da es keine Ruecksicherung gibt.
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
    """Beendete Jobs bis auf die letzten MAX_FINISHED_JOBS aus dem Speicher
    entfernen (ihr komplettes Log liegt ohnehin dauerhaft unter logs/).
    Aufruf nur mit gehaltenem JOBS_LOCK."""
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
        except Exception as e:  # run_job faengt fast alles selbst ab -- das hier ist nur das Sicherheitsnetz
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
    """Reiht einen Auftrag ein. Ist nichts anderes aktiv, startet er sofort."""
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
    """Zeigt den ctq-Befehl, den ein Auftrag mit diesen Einstellungen benutzen
    wuerde -- gebaut mit derselben Funktion wie der echte Lauf."""
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
        low_memory = False  # ohne Datei laesst sich "automatisch" nicht entscheiden
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
    """Alle aktiven und die zuletzt beendeten Auftraege (ohne Log-Zeilen)."""
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
            # Noch nicht gestartet: es gibt weder einen Prozess noch eine Ausgabedatei
            # aufzuraeumen -- einfach aus der Schlange nehmen.
            if job_id in JOB_QUEUE:
                JOB_QUEUE.remove(job_id)
            job.update({"status": "cancelled", "error": "Job abgebrochen.", "finished_at": time.time()})
            return jsonify({"ok": True})
        if status not in ("starting", "running"):
            return jsonify({"error": "Der Job laeuft nicht mehr."}), 409
        job["cancel_requested"] = True
        proc = job.get("proc")
    if proc is not None and proc.poll() is None:
        # ctq.exe ist nur ein Starter fuer einen eigenen python.exe-Kindprozess --
        # ohne /T liefe die eigentliche Quantisierung auf der GPU weiter.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            proc.kill()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Voreinstellungen: eigene Kombinationen aus Format, Layer-Schutz und
# Erweitert-Werten, dauerhaft in presets.json neben diesem Skript (config.json
# bleibt davon unberuehrt). Eingabe-/Ausgabedatei gehoeren bewusst NICHT dazu.
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
    """Nur bekannte Felder, nur gueltige Werte. Liefert (settings, None) oder (None, fehler)."""
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
    """Liest den Tensor-Header der ausgewaehlten Datei und prueft ihn gegen
    bekannte Architektur-Signaturen (siehe MODEL_DETECTION_REGISTRY). Wird
    beim Auswaehlen eines Modells im Frontend aufgerufen, nicht beim
    Auflisten aller Modelle -- deshalb spielt die Gesamtzahl der Modelle im
    Ordner fuer die Geschwindigkeit keine Rolle."""
    path = request.args.get("path", "")
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
# Einstellungen: Modell-Ordner, ComfyUI-Pfad, ctq-Override -- alles im
# Browser konfigurierbar statt hart im Code, dauerhaft in config.json.
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

    # Erst ALLES validieren, ohne CONFIG anzufassen -- sonst wuerde bei z.B.
    # gueltigen model_roots + ungueltigem comfyui_root im selben Request die
    # model_roots-Aenderung schon im Live-Server uebernommen, aber wegen des
    # Fehlers nie in config.json gespeichert (inkonsistenter Zustand, der
    # erst beim naechsten Neustart wieder verschwindet).
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
            path = (r.get("path") or "").strip().strip('"')
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
        new_comfyui_root = (data["comfyui_root"] or "").strip().strip('"')
        if new_comfyui_root and not os.path.isdir(new_comfyui_root):
            return jsonify({"error": f"ComfyUI-Ordner existiert nicht: {new_comfyui_root}"}), 400

    new_ctq_override = None
    if "ctq_path_override" in data:
        override = (data["ctq_path_override"] or "").strip().strip('"')
        if override and not os.path.isfile(override):
            return jsonify({"error": f"Datei existiert nicht: {override}"}), 400
        new_ctq_override = override or None

    # Alle Validierungen bestanden -- jetzt erst uebernehmen und speichern.
    if "model_roots" in data:
        CONFIG["model_roots"] = new_model_roots
    if "comfyui_root" in data:
        CONFIG["comfyui_root"] = new_comfyui_root
    if "ctq_path_override" in data:
        CONFIG["ctq_path_override"] = new_ctq_override
        CTQ_EXE = find_ctq_executable(new_ctq_override)

    save_config()
    _invalidate_system_check_cache()  # ComfyUI-/ctq-Pfad kann sich geaendert haben
    return jsonify({"ok": True, "model_count": len(scan_models())})


@app.route("/api/browse_native", methods=["POST"])
def api_browse_native():
    """Oeffnet einen ECHTEN Windows-Auswahldialog ueber ein eingebettetes
    PowerShell, genau wie start_server.bat es schon fuer die python.exe-
    Auswahl macht. Ein <input type=file> kann aus Sicherheitsgruenden keinen
    echten Dateisystempfad liefern, daher der Umweg ueber den Server (der
    laeuft nur auf 127.0.0.1). Blockiert den Request, bis der Nutzer
    waehlt/abbricht -- threaded=True auf app.run() sorgt dafuer, dass der
    Rest des Servers (z.B. ein laufender Quantisierungs-Job) davon
    unberuehrt bleibt.

    Fuer Ordner wird bewusst KEIN FolderBrowserDialog verwendet -- der zeigt
    unter PowerShell/WinForms den alten "Ordner suchen"-Baum-Dialog, nicht
    den modernen Explorer-Stil, den man von einem Datei-Dialog kennt (siehe
    ctq.exe/python.exe-Auswahl). Stattdessen der ueblicher Kniff: ein ganz
    normaler OpenFileDialog im modernen Stil, bei dem Dateien komplett
    ausgeblendet sind (Filter matcht nichts) und man den gewuenschten Ordner
    einfach oeffnet, waehrend ein Platzhaltertext im Dateiname-Feld stehen
    bleibt -- daraus wird danach der Ordnerpfad extrahiert.
    """
    if os.name != "nt":
        return jsonify({"error": "Nativer Dialog ist nur unter Windows verfuegbar."}), 400
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Ungueltige Anfrage."}), 400

    mode = data.get("mode", "folder")  # "folder" oder "file"
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
# Systemcheck: prueft, ob ctq und das ComfyUI-INT8-Fast-Custom-Node
# installiert sind und ob dafuer Updates verfuegbar sind.
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
    """'v1.2.0' / '1.3' -> (1, 2, 0) / (1, 3, 0); None, wenn keine Versionsnummer drinsteht."""
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(v or ""))
    if not m:
        return None
    parts = [int(x) for x in m.group(1).split(".")][:3]
    return tuple(parts + [0] * (3 - len(parts)))


def check_app_update():
    """Vergleicht APP_VERSION mit dem neuesten Release auf GitHub. Installiert wird
    nichts: der Nutzer bekommt nur den Link zum Release."""
    base = {"id": "app", "label": "GFlava-Quant", "label_en": "GFlava-Quant",
            "can_install": False, "can_update": False, "release_url": GFLAVA_RELEASES_URL}
    req = urllib.request.Request(
        f"https://api.github.com/repos/{GFLAVA_REPO}/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": f"GFlava-Quant/{APP_VERSION}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
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


# Ergebnis der letzten Pruefung. Die automatische Pruefung beim Laden der Seite nimmt es,
# solange es hoechstens eine Stunde alt ist -- sonst fragt jedes Neuladen GitHub, PyPI und
# git ab (GitHub erlaubt ohne Anmeldung nur 60 Anfragen pro Stunde).
SYSTEM_CHECK_MAX_AGE = 3600
SYSTEM_CHECK_CACHE = {"checks": None, "checked_at": 0.0}
SYSTEM_CHECK_LOCK = threading.Lock()


def _invalidate_system_check_cache():
    with SYSTEM_CHECK_LOCK:
        SYSTEM_CHECK_CACHE["checks"] = None


@app.route("/api/system_check")
def api_system_check():
    """?cached=1: automatische Pruefung, darf ein frisches Ergebnis wiederverwenden.
    Ohne: der Knopf in den Einstellungen, prueft immer neu."""
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
    _invalidate_system_check_cache()  # nach Installieren/Aktualisieren stimmt das alte Ergebnis nicht mehr

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
            continue  # z.B. zwischen Auflisten und hier geloescht
        logs.append({"name": p.name, "size": stat.st_size, "mtime": stat.st_mtime})
    return jsonify({"logs": logs})


@app.route("/api/logs/<path:name>")
def api_log_file(name):
    # Nur der Dateiname zaehlt -- verhindert Zugriff ausserhalb von logs/.
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
    # threaded=True, damit ein offener nativer Ordner-/Datei-Dialog (blockiert
    # den Request, bis der Nutzer waehlt/abbricht) nicht den ganzen Server
    # einfriert -- ein parallel laufender Quantisierungs-Job soll trotzdem
    # weiter Status liefern koennen.
    app.run(host="127.0.0.1", port=8877, debug=False, threaded=True)
