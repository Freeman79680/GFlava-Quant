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
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Konfiguration: Modell-Ordner, ComfyUI-Installation, ctq-Pfad-Override.
# Wird dauerhaft in config.json neben diesem Skript gespeichert, damit nichts
# hart im Code hinterlegt werden muss und Aenderungen einen Server-Neustart
# ueberleben. Beim allerersten Start gibt es noch keine Modell-Ordner -- die
# Einstellungen oeffnen sich dann automatisch mit einem Hinweis.
# ---------------------------------------------------------------------------

APP_VERSION = "1.2"
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


def read_safetensors_keys(path):
    """Liest nur den JSON-Header einer .safetensors-Datei (8-Byte-Laenge +
    Header), keine Tensor-Daten -- schnell auch bei sehr grossen Modellen."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        if header_size <= 0 or header_size > 200_000_000:
            raise ValueError("Ungueltiger oder zu grosser Header.")
        header = json.loads(f.read(header_size))
    return [k for k in header.keys() if k != "__metadata__"]


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
        cmd = [CTQ_EXE, "-i", input_path, "-o", output_path]
        cmd += list(fmt["flags"])
        if fmt.get("needs_groupsize"):
            cmd += ["--convrot-group-size", str(convrot_groupsize)]
        if fmt.get("needs_blocksize"):
            cmd += ["--block_size", str(block_size)]
        cmd += ["--comfy_quant", "--save-quant-metadata"]
        if resolve_low_memory_flag(low_memory_mode, input_path, log):
            cmd += ["--low-memory"]
        if exclude_arg:
            cmd += ["--exclude-layers", exclude_arg]
        if extra_builtin_flag:
            cmd += [extra_builtin_flag]
        if extra_args:
            cmd += extra_args.split()

        log("Befehl: " + " ".join(cmd))
        set_status("running", progress=0)

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


@app.route("/")
def index():
    return Response(INDEX_HTML.replace("__APP_VERSION__", APP_VERSION), mimetype="text/html")


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


@app.route("/api/quantize", methods=["POST"])
def api_quantize():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Ungueltige Anfrage."}), 400

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
        return jsonify({"error": "low_memory_mode muss 'auto', 'on' oder 'off' sein."}), 400

    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": f"Eingabedatei nicht gefunden: {input_path}"}), 400
    if not input_path.lower().endswith(".safetensors"):
        return jsonify({"error": "Die Eingabedatei muss eine .safetensors-Datei sein."}), 400
    if format_key not in FORMAT_PRESETS:
        return jsonify({"error": "Unbekanntes Quantisierungs-Format."}), 400

    try:
        convrot_groupsize = int(convrot_groupsize)
    except (TypeError, ValueError):
        return jsonify({"error": "ConvRot-Gruppengroesse muss eine Zahl sein."}), 400
    # ctq verlangt hier zwingend eine Potenz von 4 (per `ctq --help-experimental`).
    if FORMAT_PRESETS[format_key].get("needs_groupsize") and convrot_groupsize not in (4, 16, 64, 256, 1024):
        return jsonify({"error": "ConvRot-Gruppengroesse muss 4, 16, 64, 256 oder 1024 sein (Potenz von 4)."}), 400

    try:
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "Block-Groesse muss eine positive ganze Zahl sein."}), 400

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_{format_key}{ext}"

    # Sonst wuerde ctq die noch unquantisierte Originaldatei ueberschreiben --
    # unwiderruflich, da es keine Ruecksicherung gibt.
    if os.path.normcase(os.path.abspath(output_path)) == os.path.normcase(os.path.abspath(input_path)):
        return jsonify({"error": "Ausgabedatei darf nicht mit der Eingabedatei identisch sein -- "
                                  "das wuerde dein unquantisiertes Original unwiderruflich ueberschreiben."}), 400

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        return jsonify({"error": f"Zielordner existiert nicht: {out_dir}"}), 400

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

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        # Abgeschlossene Jobs wieder aus dem Speicher entfernen -- ihr
        # vollstaendiges Log liegt ohnehin schon dauerhaft unter logs/, sonst
        # wuerde JOBS bei einem lange laufenden Server unbegrenzt wachsen.
        for old_id in [k for k, v in JOBS.items() if v.get("status") in ("done", "error", "cancelled")]:
            del JOBS[old_id]
        JOBS[job_id] = {"status": "starting", "log": [], "progress": 0}

    t = threading.Thread(
        target=run_job,
        args=(job_id, input_path, output_path, format_key, exclude_arg, extra_builtin_flag,
              convrot_groupsize, block_size, extra_args, low_memory_mode),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "output_path": output_path})


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
        })


@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Unbekannte job_id."}), 404
        if job.get("status") not in ("starting", "running"):
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
        keys = read_safetensors_keys(path)
    except Exception as e:
        return jsonify({"error": f"Konnte Datei nicht lesen: {e}"}), 400
    return jsonify({"preset_key": detect_model_type(keys)})


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
                "detail": f"Installiert unter {d}. Update verfuegbar.",
                "detail_en": f"Installed at {d}. Update available."}
    return {**base, "status": "ok", "can_install": False, "can_update": False,
            "detail": f"Installiert unter {d} (aktuell).",
            "detail_en": f"Installed at {d} (up to date)."}


@app.route("/api/system_check")
def api_system_check():
    return jsonify({"checks": [check_ctq_status(), check_int8fast_status()]})


@app.route("/api/system_check/action", methods=["POST"])
def api_system_check_action():
    global CTQ_EXE
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Ungueltige Anfrage."}), 400
    check_id = data.get("id")
    action = data.get("action")

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


# ---------------------------------------------------------------------------
# Frontend (ein einziges HTML/CSS/JS-Bundle, damit alles in einer Datei bleibt)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>GFlava-Quant</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/png" href="/icon.png">
<script>
  (function () {
    try {
      if (localStorage.getItem('theme') === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
      }
      if (localStorage.getItem('minimal') === '1') {
        document.documentElement.setAttribute('data-minimal', '');
      }
    } catch (e) {}
  })();
</script>
<style>
  @font-face {
    font-family: 'Space Grotesk'; font-style: normal; font-weight: 500 700; font-display: swap;
    src: url('/fonts/space-grotesk.woff2') format('woff2');
  }
  @font-face {
    font-family: 'IBM Plex Sans'; font-style: normal; font-weight: 400 600; font-display: swap;
    src: url('/fonts/ibm-plex-sans.woff2') format('woff2');
  }
  @font-face {
    font-family: 'JetBrains Mono'; font-style: normal; font-weight: 400 600; font-display: swap;
    src: url('/fonts/jetbrains-mono.woff2') format('woff2');
  }
  :root {
    --bg: #0a0c11;
    --panel: #12151d;
    --panel2: #1a1e29;
    --panel-hover: #1f2430;
    --border: #262b3a;
    --text: #eef0f5;
    --muted: #8891a3;
    --muted-soft: #5f6879;
    --accent: #ff3ec8;
    --accent-bright: #ff7ad9;
    --accent2: #22d3ee;
    --accent3: #a855f7;
    --success: #34d399;
    --error: #f87171;
    --warn: #fbbf24;
    --radius: 18px;
    --radius-sm: 10px;
    --content-max: 1200px;
    --chip-bg: #191724;
    --chip-text: #c3c7da;
    --chip-edge-a: rgba(255,62,200,.42);
    --chip-edge-b: rgba(168,85,247,.30);
    --chip-edge-c: rgba(34,211,238,.42);
    --topbar-bg: rgba(10,11,16,.96);
    --actionbar-bg: rgba(10,11,16,.96);
    --code-bg: #090b10;
    --code-text: #b7bccb;
    --font-display: 'Space Grotesk', 'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif;
    --font-body: 'IBM Plex Sans', 'Segoe UI Variable Text', 'Segoe UI', system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif;
    --font-mono: 'JetBrains Mono', 'Cascadia Mono', Consolas, monospace;
  }
  :root[data-theme="light"] {
    --bg: #f3f4f8;
    --panel: #ffffff;
    --panel2: #eef0f5;
    --panel-hover: #e3e6ee;
    --border: #dde1ea;
    --text: #191c26;
    --muted: #5b6272;
    --muted-soft: #848da0;
    --accent: #c026d3;
    --accent-bright: #a21caf;
    --accent2: #0891b2;
    --accent3: #7c3aed;
    --success: #15803d;
    --error: #dc2626;
    --warn: #b45309;
    --chip-bg: #fbf8fe;
    --chip-text: #3b4054;
    --chip-edge-a: rgba(192,38,211,.38);
    --chip-edge-b: rgba(124,58,237,.28);
    --chip-edge-c: rgba(8,145,178,.38);
    --topbar-bg: rgba(255,255,255,.97);
    --actionbar-bg: rgba(255,255,255,.97);
    --code-bg: #eef0f5;
    --code-text: #2a2e3d;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0; color: var(--text); background: var(--bg);
    font-family: var(--font-body);
    -webkit-font-smoothing: antialiased;
  }
  /* Feststehender Hintergrund: ein Canvas, das nur bei Groessen- oder
     Theme-Wechsel neu gezeichnet wird (drawBackground). Kein Filter, keine
     3D-Transformation, keine Animation. */
  #bgCanvas {
    position: fixed; inset: 0; width: 100%; height: 100%; z-index: -1;
    pointer-events: none; display: block;
  }
  h1, h2, h3 { font-family: var(--font-display); }
  .no-transitions *, .no-transitions *::before, .no-transitions *::after { transition: none !important; }
  @keyframes pulseRing { 0% { transform: scale(.7); opacity: .55; } 100% { transform: scale(2.1); opacity: 0; } }
  @keyframes cardIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes pillPop {
    0% { opacity: 0; transform: scale(.86); }
    60% { opacity: 1; transform: scale(1.05); }
    100% { opacity: 1; transform: scale(1); }
  }
  #modelTypePill.pop { animation: pillPop .34s cubic-bezier(.16,1,.3,1); }
  @media (prefers-reduced-motion: reduce) {
    .pill.ok .dot::after, .grid-2 > .card, #modelTypePill.pop, .job-drawer.running::before {
      animation: none !important;
    }
    .modal, .chip { transition: none !important; }
  }

  /* -- Sticky Kopfzeile -- */
  .topbar {
    position: sticky; top: 0; z-index: 40;
    background: var(--topbar-bg);
    border-bottom: 1px solid var(--border);
  }
  .topbar-inner {
    max-width: var(--content-max); margin: 0 auto; padding: 18px 32px;
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    flex-wrap: wrap; row-gap: 12px;
  }
  .brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
  .brand-icon {
    position: relative; width: 40px; height: 40px; flex: none;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(212,175,106,.38) 0%, rgba(212,175,106,.12) 55%, transparent 72%);
  }
  .brand-icon img { width: 40px; height: 40px; object-fit: contain; }
  .brand-text { min-width: 0; }
  h1 {
    display: flex; align-items: center; gap: 9px;
    font-size: 1.19rem; margin: 0; letter-spacing: -.01em; font-weight: 700; line-height: 1.2;
  }
  .version-chip {
    padding: 3px 9px; border-radius: 999px; flex: none;
    background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
    font-family: var(--font-mono); font-size: 0.66rem; font-weight: 500; letter-spacing: .04em;
  }
  .brand-sub {
    display: block; color: var(--muted-soft); font-size: 0.75rem; margin-top: 3px;
    font-family: var(--font-mono); letter-spacing: .01em;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .topbar-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .lang-select {
    width: auto; height: 38px; padding: 0 12px; border-radius: 11px; font-size: 0.8rem;
    background: var(--panel2); border: 1px solid var(--border); color: var(--text);
  }
  .topbar-controls button.icon-btn {
    width: 38px; height: 38px; padding: 0; border-radius: 11px;
    display: flex; align-items: center; justify-content: center; color: var(--text);
  }
  .topbar-controls button.icon-btn svg { width: 17px; height: 17px; }
  #ctqStatus { height: 38px; padding: 0 14px; font-family: var(--font-mono); font-size: 0.75rem; gap: 9px; }
  #ctqStatus.missing { cursor: pointer; }
  :root:not([data-theme="light"]) .icon-moon { display: none; }
  :root[data-theme="light"] .icon-sun { display: none; }
  .pill {
    display: inline-flex; align-items: center; gap: 7px; flex: none;
    font-size: 0.78rem; padding: 6px 12px; border-radius: 999px;
    background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
    max-width: 46vw;
  }
  .pill span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .pill .dot { position: relative; width: 7px; height: 7px; border-radius: 50%; background: var(--muted-soft); flex: none; }
  .pill.ok { color: var(--success); border-color: rgba(52,211,153,.25); background: rgba(52,211,153,.08); }
  .pill.ok .dot { background: var(--success); box-shadow: 0 0 0 3px rgba(52,211,153,.18); }
  .pill.ok .dot::after {
    content: ''; position: absolute; inset: -4px; border-radius: 50%;
    background: var(--success); opacity: .45; animation: pulseRing 2.2s ease-out infinite;
  }
  .pill.missing { color: var(--error); border-color: rgba(248,113,113,.25); background: rgba(248,113,113,.08); }
  .pill.missing .dot { background: var(--error); box-shadow: 0 0 0 3px rgba(248,113,113,.18); }
  #modelTypePill { max-width: 100%; }
  #modelTypePill.loading { color: var(--muted); }
  #modelTypePill.loading .dot { background: var(--muted-soft); animation: pulse 1.4s ease-in-out infinite; }
  #modelTypePill.type-anima { color: #a78bfa; border-color: rgba(167,139,250,.3); background: rgba(167,139,250,.12); }
  #modelTypePill.type-anima .dot { background: #a78bfa; }
  #modelTypePill.type-flux2 { color: #fb923c; border-color: rgba(251,146,60,.3); background: rgba(251,146,60,.12); }
  #modelTypePill.type-flux2 .dot { background: #fb923c; }
  #modelTypePill.type-flux1 { color: #f87171; border-color: rgba(248,113,113,.3); background: rgba(248,113,113,.12); }
  #modelTypePill.type-flux1 .dot { background: #f87171; }
  #modelTypePill.type-sdxl { color: #4ade80; border-color: rgba(74,222,128,.3); background: rgba(74,222,128,.12); }
  #modelTypePill.type-sdxl .dot { background: #4ade80; }
  #modelTypePill.type-qwen { color: #2dd4bf; border-color: rgba(45,212,191,.3); background: rgba(45,212,191,.12); }
  #modelTypePill.type-qwen .dot { background: #2dd4bf; }
  #modelTypePill.type-zimage { color: #f472b6; border-color: rgba(244,114,182,.3); background: rgba(244,114,182,.12); }
  #modelTypePill.type-zimage .dot { background: #f472b6; }
  #modelTypePill.type-video { color: #60a5fa; border-color: rgba(96,165,250,.3); background: rgba(96,165,250,.12); }
  #modelTypePill.type-video .dot { background: #60a5fa; }
  #modelTypePill.type-other { color: #facc15; border-color: rgba(250,204,21,.3); background: rgba(250,204,21,.12); }
  #modelTypePill.type-other .dot { background: #facc15; }

  /* -- Hauptinhalt -- */
  .wrap { max-width: var(--content-max); margin: 0 auto; padding: 28px 32px 150px; }
  .page-meta {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 4px 0 22px;
  }
  #statsBar {
    color: var(--muted-soft); font-size: 0.78rem;
    font-family: var(--font-mono); letter-spacing: .02em;
  }

  .intro-details {
    background: var(--panel2); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px 22px; margin-bottom: 26px;
  }
  .intro-details summary { color: var(--text); font-weight: 600; font-size: 0.92rem; }
  .intro-details[open] summary { margin-bottom: 12px; }
  .intro-details p { margin: 0; color: var(--muted); line-height: 1.65; font-size: 0.9rem; max-width: 74ch; }

  .card {
    position: relative; overflow: hidden;
    background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0) 140px), var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius); padding: 28px 30px; margin-bottom: 24px;
  }
  .card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent3), var(--accent2));
  }
  .card h2 {
    font-size: 1.02rem; margin: 0 0 20px; color: var(--text); font-weight: 700;
    letter-spacing: -.01em;
  }
  .step-head { display: flex; align-items: center; gap: 13px; margin-bottom: 20px; }
  .step-head h2 { margin: 0; }
  .step-num {
    width: 30px; height: 30px; border-radius: 50%; flex: none;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff; font-weight: 700; font-size: 0.85rem;
    font-family: var(--font-mono);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px -3px rgba(255,62,200,.55);
  }
  #advancedDetails > summary, #advancedDetails > summary > * { font-size: 0.95rem; font-weight: 600; color: var(--text); }

  label { display: block; font-size: 0.88rem; margin-bottom: 7px; color: var(--text); font-weight: 500; }
  input[type=text], input[type=number], select, textarea {
    width: 100%; background: var(--panel2); border: 1px solid var(--border);
    color: var(--text); padding: 11px 14px; border-radius: var(--radius-sm); font-size: 0.93rem;
    font-family: inherit; transition: border-color .15s, box-shadow .15s;
  }
  input::placeholder { color: var(--muted-soft); }
  input:hover, select:hover { border-color: #384056; }
  input:focus, select:focus, textarea:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(255,62,200,.22);
  }
  select { cursor: pointer; }
  .row { margin-bottom: 20px; }
  .row:last-child { margin-bottom: 0; }
  .hint { color: var(--muted); font-size: 0.81rem; margin-top: 7px; line-height: 1.6; }
  .badge { display: inline-block; font-size: 0.68rem; padding: 2px 8px; border-radius: 999px; margin-left: 6px; font-weight: 600; letter-spacing: .02em; vertical-align: 1px; }
  .badge.ok { background: rgba(52,211,153,.14); color: var(--success); }
  .badge.warn { background: rgba(251,191,36,.14); color: var(--warn); }
  .badge.info { background: rgba(255,62,200,.16); color: var(--accent-bright); }
  button {
    background: linear-gradient(135deg, var(--accent), var(--accent3));
    color: #fff; border: none; font-weight: 600;
    padding: 13px 20px; border-radius: var(--radius-sm); font-size: 0.95rem; cursor: pointer;
    box-shadow: 0 6px 20px -8px rgba(255,62,200,.55);
    transition: transform .12s, box-shadow .12s, opacity .12s, background-color .12s, border-color .12s, color .12s;
  }
  button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 10px 24px -8px rgba(255,62,200,.7); }
  button:active:not(:disabled) { transform: translateY(0); }
  button:disabled { opacity: 0.5; cursor: not-allowed; box-shadow: none; }
  button.secondary {
    background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    box-shadow: none; font-weight: 500;
  }
  button.secondary:hover:not(:disabled) { background: var(--panel-hover); transform: none; box-shadow: none; border-color: rgba(255,62,200,.45); }
  button.icon-btn { padding: 10px 13px; flex: none; }
  button.link-btn {
    background: none; border: none; color: var(--accent-bright); box-shadow: none;
    font-size: 0.83rem; font-weight: 500; padding: 2px 0; text-decoration: underline;
    text-underline-offset: 3px; text-decoration-color: transparent; transition: text-decoration-color .15s;
  }
  button.link-btn:hover { text-decoration-color: var(--accent-bright); transform: none; box-shadow: none; }
  /* -- Modell-Suche (Combobox mit Echtzeit-Filter) -- */
  .combo { position: relative; }
  .combo-row { display: flex; gap: 8px; }
  .combo-panel {
    position: absolute; top: calc(100% + 6px); left: 0; right: 0; z-index: 30;
    background: var(--panel2); border: 1px solid var(--border); border-radius: var(--radius-sm);
    max-height: 320px; overflow-y: auto; padding: 6px;
  }
  .combo-panel::-webkit-scrollbar { width: 8px; }
  .combo-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }
  .combo-group-label {
    padding: 8px 10px 4px; font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; color: var(--muted-soft); font-family: var(--font-mono);
  }
  .combo-item {
    padding: 9px 10px; border-radius: 7px; cursor: pointer; font-size: 0.87rem; color: var(--text);
    display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
  }
  .combo-item:hover, .combo-item.active { background: rgba(255,62,200,.14); }
  .combo-item .combo-item-size {
    color: var(--muted); font-size: 0.76rem; flex: none;
    font-family: var(--font-mono);
  }
  .combo-item mark { background: rgba(255,62,200,.4); color: var(--accent-bright); border-radius: 3px; padding: 0 1px; }
  .combo-empty { padding: 16px 10px; color: var(--muted); font-size: 0.85rem; text-align: center; }
  /* Chips: Verlaufs-Rahmen ueber zwei Hintergrund-Ebenen (padding-box/border-box)
     statt Schatten -- kostet nicht mehr als eine normale Hintergrundfarbe. */
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
  .chip {
    --chip-edge: linear-gradient(120deg, var(--chip-edge-a), var(--chip-edge-b) 50%, var(--chip-edge-c));
    background: linear-gradient(var(--chip-bg), var(--chip-bg)) padding-box, var(--chip-edge) border-box;
    border: 1px solid transparent; box-shadow: none;
    color: var(--chip-text); font-family: var(--font-mono); font-size: 0.76rem; font-weight: 500;
    padding: 7px 13px; border-radius: 999px; cursor: pointer;
    transition: transform .12s ease, color .12s ease;
  }
  .chip:hover:not(:disabled) {
    --chip-edge: linear-gradient(120deg, var(--accent), var(--accent3) 50%, var(--accent2));
    color: var(--text); transform: translateY(-1px); box-shadow: none;
  }
  .chip.active, .chip.active:hover:not(:disabled) {
    background: linear-gradient(120deg, var(--accent), var(--accent3)) border-box;
    color: #fff; font-weight: 600;
  }
  details summary {
    cursor: pointer; color: var(--muted); font-size: 0.88rem; margin-bottom: 4px;
    list-style: none; display: flex; align-items: center; gap: 7px; user-select: none;
  }
  details summary::-webkit-details-marker { display: none; }
  details summary::before { content: '\25B8'; font-size: .7rem; transition: transform .15s; color: var(--muted-soft); flex: none; }
  details[open] summary::before { transform: rotate(90deg); }
  details[open] summary { margin-bottom: 18px; }
  .progress-outer { height: 8px; border-radius: 5px; background: var(--panel2); overflow: hidden; margin: 12px 0; }
  .progress-inner {
    height: 100%; width: 0%; border-radius: 5px; transition: width .3s;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    background-size: 200% 100%;
  }
  .progress-inner.running { animation: shimmer 1.4s linear infinite; }
  @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
  #log, #ctqHelpOutput, #logViewer {
    background: var(--code-bg); border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 12px 14px; overflow-y: auto; overflow-x: hidden; box-sizing: border-box;
    font-family: var(--font-mono), "SFMono-Regular";
    font-size: 0.78rem; white-space: pre-wrap; word-break: break-all;
    overflow-wrap: anywhere; color: var(--code-text); line-height: 1.5;
  }
  #log { height: 220px; }
  #ctqHelpOutput, #logViewer { max-height: 340px; margin-top: 10px; }
  #log::-webkit-scrollbar, #ctqHelpOutput::-webkit-scrollbar, #logViewer::-webkit-scrollbar { width: 9px; }
  #log::-webkit-scrollbar-thumb, #ctqHelpOutput::-webkit-scrollbar-thumb, #logViewer::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }
  #log::-webkit-scrollbar-track, #ctqHelpOutput::-webkit-scrollbar-track, #logViewer::-webkit-scrollbar-track { background: transparent; }
  #jobLogLink { display: none; margin: 6px 0; overflow-wrap: anywhere; word-break: break-all; }
  #jobLogLink a { color: var(--accent-bright); overflow-wrap: anywhere; word-break: break-all; }
  optgroup { color: var(--muted); font-weight: 600; }
  option { color: var(--text); font-weight: 400; }
  code {
    background: var(--panel2); padding: 2px 6px; border-radius: 5px;
    font-family: var(--font-mono); font-size: 0.85em; color: var(--accent-bright);
  }
  .logs-list { display: flex; flex-direction: column; gap: 7px; }
  .log-row {
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
    background: var(--panel2); border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 10px 14px; font-size: 0.82rem;
  }
  .log-row span { color: var(--muted); font-family: var(--font-mono); font-size: 0.78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .log-row button { padding: 6px 12px; font-size: 0.8rem; }

  /* -- Responsives Grid statt alles untereinander, gleich hohe Karten -- */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-bottom: 24px; }
  .grid-2 > .card { margin-bottom: 0; animation: cardIn .5s cubic-bezier(.16,1,.3,1) backwards; }
  .grid-2 > .card:nth-child(2) { animation-delay: .08s; }
  @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }

  /* -- Fixierte Action-Leiste: der Start-Button ist immer erreichbar -- */
  .action-bar {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 45;
    background: var(--actionbar-bg);
    border-top: 1px solid var(--border);
  }
  .action-bar-inner {
    max-width: var(--content-max); margin: 0 auto; padding: 16px 32px;
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
  }
  .action-summary { color: var(--muted); font-size: 0.84rem; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .action-summary b { color: var(--text); font-weight: 600; }
  .action-bar #runBtn { width: auto; min-width: 250px; padding: 13px 30px; flex: none; font-size: 0.96rem; }
  .action-progress {
    display: none; flex-direction: column; gap: 4px; flex: 1 1 220px; max-width: 320px; min-width: 160px;
  }
  .action-progress .progress-outer { margin: 0; }
  .action-progress .job-drawer-timer { margin-top: 0; }

  /* -- Job-Flyout: oeffnet mittig mit 1000x500, Verlaufsrahmen rundum + Schatten.
     Rahmen = Verlaufsflaeche (::before) hinter dem Inhalt, darueber die Panel-
     flaeche (::after) 2px kleiner. Waehrend eines Jobs dreht sich nur der
     Verlaufswinkel auf eigener Ebene -- der Log-Inhalt wird nicht neu gemalt. -- */
  @property --drawer-angle { syntax: '<angle>'; inherits: false; initial-value: 0deg; }
  .job-drawer {
    position: fixed; left: 50%; top: 50%; width: 1000px; height: 500px;
    min-width: 300px; min-height: 180px;
    max-width: calc(100vw - 32px); max-height: calc(100vh - 110px);
    display: flex; flex-direction: column; overflow: hidden; z-index: 50; isolation: isolate;
    border-radius: var(--radius); padding: 24px; box-sizing: border-box;
    box-shadow: 0 12px 32px rgba(0,0,0,.6), 0 2px 8px rgba(0,0,0,.45);
    transform: translate(-50%, -48%) scale(.97); opacity: 0; pointer-events: none;
    transition: transform .3s cubic-bezier(.16,1,.3,1), opacity .22s ease;
  }
  :root[data-theme="light"] .job-drawer {
    box-shadow: 0 12px 32px rgba(30,34,60,.24), 0 2px 8px rgba(30,34,60,.16);
  }
  .job-drawer.open { transform: translate(-50%, -50%); opacity: 1; pointer-events: auto; }
  .job-drawer.placed { transform: scale(.97); }
  .job-drawer.placed.open { transform: none; }
  .job-drawer::before {
    content: ''; position: absolute; inset: 0; z-index: -2; border-radius: inherit;
    background: conic-gradient(from var(--drawer-angle), var(--accent), var(--accent3), var(--accent2), var(--accent3), var(--accent));
  }
  .job-drawer::after {
    content: ''; position: absolute; inset: 2px; z-index: -1;
    border-radius: calc(var(--radius) - 2px); background: var(--panel);
  }
  .job-drawer.running::before { will-change: transform; animation: drawerBorderSpin 4s linear infinite; }
  @keyframes drawerBorderSpin { to { --drawer-angle: 360deg; } }
  .job-drawer.no-anim { transition: none; }
  .job-drawer-header {
    display: flex; align-items: flex-start; justify-content: space-between; gap: 10px;
    margin-bottom: 6px; flex: none; cursor: move; user-select: none;
  }
  .job-drawer-header-text { min-width: 0; flex: 1 1 auto; }
  .job-drawer-header .icon-btn { flex: none; cursor: pointer; }
  .job-drawer-status { font-family: var(--font-display); font-size: 0.95rem; font-weight: 600; overflow-wrap: anywhere; word-break: break-word; }
  .job-drawer-status.status-running { color: var(--accent-bright); }
  .job-drawer-status.status-done { color: var(--success); }
  .job-drawer-status.status-error { color: var(--error); }
  .job-drawer-status.status-cancelled { color: var(--warn); }
  .drawer-reopen-pill.status-cancelled .dot { background: var(--warn); animation: none; }
  .job-drawer-actions { display: flex; align-items: center; gap: 8px; flex: none; }
  .job-drawer-actions .cancel-job-btn { padding: 7px 14px; font-size: 0.8rem; }
  .action-buttons { display: flex; align-items: center; gap: 10px; flex: none; }
  .action-bar .cancel-job-btn { padding: 13px 20px; font-size: 0.9rem; }
  .cancel-job-btn { display: none; }
  body.job-active .cancel-job-btn { display: inline-flex; align-items: center; }
  button.cancel-job-btn:hover:not(:disabled) { border-color: var(--error); color: var(--error); }
  .job-drawer-timer {
    color: var(--muted); font-size: 0.78rem; margin-top: 2px;
    font-family: var(--font-mono);
  }
  .job-drawer #log { flex: 1 1 auto; height: auto; min-height: 60px; }
  #jobLogLink { flex: none; }
  .job-drawer-resize-handle {
    position: absolute; right: 3px; bottom: 3px; width: 18px; height: 18px;
    cursor: nwse-resize; opacity: .45; z-index: 2;
  }
  .job-drawer-resize-handle:hover { opacity: .9; }
  .job-drawer-resize-handle::before {
    content: ''; position: absolute; right: 3px; bottom: 3px; width: 9px; height: 9px;
    border-right: 2px solid var(--muted-soft); border-bottom: 2px solid var(--muted-soft);
  }
  body.drawer-noselect { user-select: none; }
  .drawer-reopen-pill {
    position: fixed; top: 84px; right: 24px; z-index: 49;
    display: none; align-items: center; gap: 9px;
    background: var(--panel); border: 1px solid var(--border); color: var(--text);
    padding: 10px 16px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;
    cursor: pointer; box-shadow: none;
  }
  .drawer-reopen-pill:hover:not(:disabled) { box-shadow: none; }
  .drawer-reopen-pill .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-bright); flex: none; animation: pulse 1.4s ease-in-out infinite; }
  .drawer-reopen-pill.status-done .dot { background: var(--success); animation: none; }
  .drawer-reopen-pill.status-error .dot { background: var(--error); animation: none; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }

  /* -- Einstellungen: Dropdown unter dem Zahnrad, ohne Abdunkeln -- */
  .settings-anchor { position: relative; }
  .topbar.settings-open { z-index: 60; }
  .settings-dropdown {
    position: absolute; top: calc(100% + 10px); right: 0; z-index: 80;
    width: min(560px, calc(100vw - 32px));
    opacity: 0; visibility: hidden; transform: translateY(-6px); pointer-events: none;
    transition: opacity .14s ease, transform .18s cubic-bezier(.16,1,.3,1), visibility 0s linear .18s;
  }
  .settings-dropdown.open {
    opacity: 1; visibility: visible; transform: none; pointer-events: auto;
    transition: opacity .14s ease, transform .18s cubic-bezier(.16,1,.3,1), visibility 0s;
  }
  #settingsBtn[aria-expanded="true"] { border-color: var(--accent); color: var(--accent-bright); }
  .modal {
    position: relative; overflow-x: hidden; overflow-y: auto; overscroll-behavior: contain;
    max-height: calc(100vh - 110px);
    background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
    padding: 24px 24px 22px;
  }
  .modal::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent3), var(--accent2));
  }
  .modal-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
  .modal-header h2 { margin: 0; font-size: 1.19rem; font-weight: 700; }
  .modal .modal-header button.icon-btn {
    width: 30px; height: 30px; padding: 0; border-radius: 9px; font-size: 1.1rem; line-height: 1;
    display: flex; align-items: center; justify-content: center;
    background: var(--panel2); color: var(--muted);
  }
  .modal > .modal-hint { margin: 0 0 22px; max-width: 52ch; }
  .modal-section { margin-top: 0; margin-bottom: 24px; }
  .modal-section h3 {
    font-family: var(--font-mono);
    font-size: 0.69rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted-soft);
    margin: 0 0 12px; font-weight: 700;
  }
  .modal-hint { color: var(--muted); font-size: 0.78rem; line-height: 1.6; margin: -4px 0 12px; }

  /* Kompakte Buttons im Modal: Ghost (sekundaer) und Verlauf (primaer) */
  .modal button {
    padding: 8px 15px; border-radius: 9px; font-size: 0.75rem; font-weight: 600;
    box-shadow: none; transition: background .1s, border-color .1s, color .1s;
  }
  .modal button:hover:not(:disabled) { transform: none; box-shadow: none; opacity: .9; }
  .modal button.secondary {
    background: transparent; border: 1px solid var(--border); color: var(--text); font-weight: 500;
  }
  .modal button.secondary:hover:not(:disabled) { background: var(--panel2); border-color: rgba(255,62,200,.45); opacity: 1; }
  .modal input[type=text] { font-size: 0.82rem; padding: 10px 12px; }

  .root-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 10px; }
  .root-row {
    display: flex; align-items: center; gap: 12px;
    background: var(--panel2); border: 1px solid var(--border); border-radius: 12px;
    padding: 10px 12px;
  }
  .root-row-icon { color: var(--accent2); flex: none; display: flex; }
  .root-row-text { min-width: 0; flex: 1; }
  .root-row-label {
    font-size: 0.82rem; font-weight: 600; color: var(--text); margin-bottom: 2px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .root-row-path {
    font-size: 0.72rem; color: var(--muted); font-family: var(--font-mono);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .modal .root-row button.icon-btn {
    width: 26px; height: 26px; padding: 0; border-radius: 7px; flex: none; font-size: 1rem; line-height: 1;
    display: flex; align-items: center; justify-content: center;
    background: transparent; color: var(--muted);
  }
  .modal .root-row button.icon-btn:hover:not(:disabled) { color: var(--error); border-color: var(--error); background: transparent; }
  .modal button#addRootBtn {
    display: flex; align-items: center; gap: 8px; width: 100%;
    padding: 10px 12px; border-radius: 12px; border: 1px dashed var(--border);
    background: transparent; color: var(--muted); font-weight: 500; font-size: 0.78rem;
  }
  .modal button#addRootBtn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent-bright); background: transparent; }

  .field-with-browse { display: flex; gap: 8px; }
  .field-with-browse input { flex: 1; }
  .field-with-browse button { flex: none; }
  .check-row {
    display: flex; align-items: center; gap: 14px;
    background: var(--panel2); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 16px; margin-bottom: 10px;
  }
  .check-icon {
    width: 34px; height: 34px; border-radius: 10px; flex: none;
    display: flex; align-items: center; justify-content: center;
  }
  .check-icon.ok { background: rgba(52,211,153,.14); color: var(--success); }
  .check-icon.warn { background: rgba(251,191,36,.14); color: var(--warn); }
  .check-icon.missing { background: rgba(248,113,113,.14); color: var(--error); }
  .check-icon.unknown { background: rgba(136,145,163,.14); color: var(--muted-soft); }
  .check-row-text { min-width: 0; flex: 1; }
  .check-row-label { font-size: 0.82rem; font-weight: 600; color: var(--text); }
  .check-row-detail {
    font-size: 0.72rem; color: var(--muted); margin-top: 3px; line-height: 1.5;
    font-family: var(--font-mono); overflow-wrap: anywhere;
  }
  .check-row-actions { display: flex; gap: 8px; flex: none; }
  .modal-actions {
    display: flex; justify-content: flex-end; gap: 10px;
    margin-top: 8px; padding-top: 20px; border-top: 1px solid var(--border);
  }
  .modal-actions button { padding: 10px 18px; font-size: 0.82rem; }

  /* -- Minimal-Design: eine Akzentfarbe, keine Verlaeufe, keine Leuchteffekte.
     Alle Verlaeufe laufen ueber --accent/--accent2/--accent3 -- sind die drei
     gleich, werden sie automatisch flach. -- */
  #minimalToggleBtn[aria-pressed="true"] { border-color: var(--accent); color: var(--accent-bright); }
  :root:not([data-minimal]) #minimalToggleBtn .icon-full { display: none; }
  :root[data-minimal] #minimalToggleBtn .icon-minimal { display: none; }
  :root[data-minimal] { --accent2: var(--accent); --accent3: var(--accent); }
  :root[data-minimal] .brand-icon { background: none; }
  :root[data-minimal] .card { background: var(--panel); }
  :root[data-minimal] button,
  :root[data-minimal] button:hover:not(:disabled),
  :root[data-minimal] .step-num { box-shadow: none; }
  :root[data-minimal] .chip {
    --chip-bg: var(--panel2);
    --chip-edge: linear-gradient(var(--border), var(--border));
  }
  :root[data-minimal] .chip:hover:not(:disabled) { --chip-edge: linear-gradient(var(--accent), var(--accent)); }
  :root[data-minimal] #modelTypePill:not(.loading) {
    color: var(--accent-bright); border-color: var(--border); background: var(--panel2);
  }
  :root[data-minimal] #modelTypePill:not(.loading) .dot { background: var(--accent); }
  :root[data-minimal] .progress-inner.running,
  :root[data-minimal] .job-drawer.running::before { animation: none; }

  @media (max-width: 640px) {
    .topbar-inner { padding: 12px 18px; }
    .pill { max-width: 100%; }
    .wrap { padding: 22px 18px 190px; }
    .card { padding: 22px 20px; }
    .action-bar-inner { padding: 12px 18px; flex-direction: column; align-items: stretch; gap: 10px; }
    .action-summary { text-align: center; white-space: normal; }
    .action-progress { max-width: 100%; }
    .action-bar #runBtn { width: 100%; min-width: 0; flex: 1; }
    .action-buttons { width: 100%; }
    .job-drawer, .drawer-reopen-pill { left: 16px; right: 16px; top: 76px; width: auto; }
    .job-drawer { max-height: calc(100vh - 240px); height: auto; transform: translateY(-8px) scale(.98); }
    .job-drawer.open, .job-drawer.placed.open { transform: none; }
    .job-drawer-resize-handle { display: none; }
    .job-drawer-header { cursor: default; }
    .drawer-reopen-pill { justify-content: center; }
    .modal { padding: 20px 18px 20px; }
    .settings-dropdown { position: fixed; top: 70px; left: 12px; right: 12px; width: auto; }
  }
  @media (max-width: 480px) {
    h1 { font-size: 1.05rem; }
    .brand-sub { display: none; }
  }
</style>
</head>
<body>
<canvas id="bgCanvas" aria-hidden="true"></canvas>
<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <div class="brand-icon">
        <img src="/icon.png" alt="GFlava-Quant">
      </div>
      <div class="brand-text">
        <h1><span data-i18n="title"></span><span class="version-chip">v__APP_VERSION__</span></h1>
        <span class="brand-sub" data-i18n="brandSub"></span>
      </div>
    </div>
    <div class="topbar-controls">
      <div class="pill" id="ctqStatus"><span class="dot"></span><span id="ctqStatusText">ctq ...</span></div>
      <select id="langSelect" class="lang-select" aria-label="Language / Sprache">
        <option value="de">Deutsch</option>
        <option value="en">English</option>
      </select>
      <button type="button" class="secondary icon-btn" id="themeToggleBtn" title="Theme" aria-label="Toggle dark/light theme">
        <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
        <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"/></svg>
      </button>
      <div class="settings-anchor">
      <button type="button" class="secondary icon-btn" id="settingsBtn" data-i18n-title="settingsTitle" aria-label="Settings" aria-haspopup="true" aria-expanded="false">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:16px;height:16px;"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
      </button>
      <div class="settings-dropdown" id="settingsOverlay" role="dialog" aria-labelledby="settingsHeadingEl">
        <div class="modal">
          <div class="modal-header">
            <h2 id="settingsHeadingEl" data-i18n="settingsHeading"></h2>
            <button type="button" class="secondary icon-btn" id="settingsCloseBtn" data-i18n-title="closeTitle">&times;</button>
          </div>
          <p class="modal-hint" data-i18n="settingsIntro"></p>

          <div class="modal-section">
            <h3 data-i18n="settingsModelRootsHeading"></h3>
            <div class="root-list" id="settingsRootList"></div>
            <button type="button" class="secondary" id="addRootBtn" data-i18n="settingsAddRoot"></button>
          </div>

          <div class="modal-section">
            <h3 data-i18n="settingsComfyuiHeading"></h3>
            <p class="modal-hint" data-i18n="settingsComfyuiHint"></p>
            <div class="field-with-browse">
              <input type="text" id="comfyuiRootInput" data-i18n-ph="settingsComfyuiPh">
              <button type="button" class="secondary" id="browseComfyuiBtn" data-i18n="settingsBrowse"></button>
            </div>
          </div>

          <div class="modal-section">
            <h3 data-i18n="settingsCtqHeading"></h3>
            <p class="modal-hint" id="settingsCtqAutoHint"></p>
            <div class="field-with-browse">
              <input type="text" id="ctqOverrideInput" data-i18n-ph="settingsCtqPh">
              <button type="button" class="secondary" id="browseCtqBtn" data-i18n="settingsBrowse"></button>
            </div>
          </div>

          <div class="modal-section">
            <h3 data-i18n="settingsCheckHeading"></h3>
            <p class="modal-hint" data-i18n="settingsCheckHint"></p>
            <div id="systemCheckList"></div>
            <button type="button" class="secondary" id="runSystemCheckBtn" data-i18n="settingsRunCheck"></button>
          </div>

          <div id="settingsError" class="hint" style="display:none; color: var(--error); margin-top:16px;"></div>
          <div class="modal-actions">
            <button type="button" class="secondary" id="settingsCancelBtn" data-i18n="settingsCancel"></button>
            <button type="button" id="settingsSaveBtn" data-i18n="settingsSave"></button>
          </div>
        </div>
      </div>
      </div>
      <button type="button" class="secondary icon-btn" id="minimalToggleBtn" aria-pressed="false">
        <svg class="icon-minimal" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="3"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
        <svg class="icon-full" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/></svg>
      </button>
    </div>
  </div>
</header>

<main class="wrap">
  <div class="page-meta">
    <div id="statsBar"></div>
  </div>

  <details class="intro-details">
    <summary data-i18n="introSummary"></summary>
    <p data-i18n="introText"></p>
  </details>

  <div class="grid-2">
  <div class="card">
    <div class="step-head"><span class="step-num">1</span><h2 data-i18n="stepModelTitle"></h2></div>
    <div class="row">
      <label data-i18n="modelSelectLabel"></label>
      <div class="combo" id="modelCombo">
        <div class="combo-row">
          <input type="text" id="modelSearch" data-i18n-ph="modelSearchPh" autocomplete="off">
          <button type="button" class="secondary icon-btn" id="refreshModelsBtn" data-i18n-title="refreshModelsTitle">&#8635;</button>
        </div>
        <div class="combo-panel" id="modelComboPanel" style="display:none;"></div>
      </div>
      <div class="hint" id="modelSelectHint"></div>
      <div class="hint" id="modelPathPreview" style="display:none;"></div>
      <div id="modelTypePill" class="badge detected" style="display:none; margin-top:8px;"></div>
    </div>
    <div class="row">
      <button type="button" class="link-btn" id="manualPathToggle"></button>
    </div>
    <div class="row" id="manualPathRow" style="display:none;">
      <label data-i18n="manualPathLabel"></label>
      <input type="text" id="inputPath" data-i18n-ph="manualPathPh">
      <div class="hint" data-i18n="manualPathHint"></div>
    </div>
    <div class="row">
      <label data-i18n="outputPathLabel"></label>
      <input type="text" id="outputPath" data-i18n-ph="outputPathPh">
      <div class="hint" data-i18n="outputPathHint"></div>
    </div>
  </div>

  <div class="card">
    <div class="step-head"><span class="step-num">2</span><h2 data-i18n="stepFormatTitle"></h2></div>
    <div class="row">
      <select id="formatSelect"></select>
      <div class="hint" id="formatHint"></div>
    </div>
    <div class="row" id="groupsizeRow">
      <label data-i18n="groupsizeLabel"></label>
      <input type="number" id="convrotGroupsize" value="256" step="1">
      <div class="hint" data-i18n="groupsizeHint"></div>
    </div>
    <div class="row" id="blockSizeRow" style="display:none;">
      <label data-i18n="blockSizeLabel"></label>
      <input type="number" id="blockSize" value="128" step="1" min="1">
      <div class="hint" data-i18n="blockSizeHint"></div>
    </div>
  </div>
  </div>

  <div class="card">
    <div class="step-head"><span class="step-num">3</span><h2 data-i18n="stepExcludeTitle"></h2></div>
    <p class="hint" style="margin-top:-4px;" data-i18n="excludeIntro"></p>
    <div class="row">
      <div class="chips" id="presetChips"></div>
      <select id="excludeSelect"></select>
      <div class="hint" id="excludeHint"></div>
    </div>
    <div class="row" id="customRegexRow" style="display:none;">
      <label data-i18n="customRegexLabel"></label>
      <input type="text" id="customRegex" placeholder="(img_in|txt_in|modulation|...)">
      <div class="hint" data-i18n="customRegexHint"></div>
    </div>
  </div>

  <div class="card">
    <details id="advancedDetails">
      <summary data-i18n="advancedSummary"></summary>
      <div class="hint" id="advancedPresetHint" style="display:none; margin-bottom:14px;"></div>
      <div class="row">
        <label data-i18n="lowMemoryLabel"></label>
        <select id="lowMemoryMode">
          <option value="auto" data-i18n="lowMemoryAuto"></option>
          <option value="on" data-i18n="lowMemoryOn"></option>
          <option value="off" data-i18n="lowMemoryOff"></option>
        </select>
        <div class="hint" data-i18n="lowMemoryHint"></div>
      </div>
      <div class="row">
        <label data-i18n="extraArgsLabel"></label>
        <input type="text" id="extraArgs" data-i18n-ph="extraArgsPh">
        <div class="hint" data-i18n="extraArgsHint"></div>
        <div class="row" style="margin-top:10px;">
          <button type="button" class="secondary" id="ctqHelpBtn" data-i18n="ctqHelpShow"></button>
        </div>
        <pre id="ctqHelpOutput" style="display:none;"></pre>
      </div>
    </details>
  </div>

  <div class="card">
    <h2 data-i18n="logsTitle"></h2>
    <p class="hint" style="margin-top:-4px;" data-i18n="logsIntro"></p>
    <div class="logs-list" id="logsList"></div>
    <pre id="logViewer" style="display:none;"></pre>
  </div>
</main>

<div class="action-bar">
  <div class="action-bar-inner">
    <div class="action-summary" id="actionSummary"></div>
    <div class="action-progress" id="actionProgress">
      <div class="progress-outer"><div class="progress-inner" id="progressBar"></div></div>
      <div id="drawerTimer" class="job-drawer-timer"></div>
    </div>
    <div class="action-buttons">
      <button type="button" class="secondary cancel-job-btn" id="cancelJobBarBtn" data-i18n="cancelJobBtn"></button>
      <button id="runBtn" data-i18n="runBtn"></button>
    </div>
  </div>
</div>

<div class="job-drawer" id="jobDrawer">
  <div class="job-drawer-header" id="jobDrawerHeader">
    <div class="job-drawer-header-text">
      <div id="drawerStatusLine" class="job-drawer-status"></div>
    </div>
    <div class="job-drawer-actions">
      <button type="button" class="secondary cancel-job-btn" id="cancelJobDrawerBtn" data-i18n="cancelJobBtn"></button>
      <button type="button" class="secondary icon-btn" id="drawerCloseBtn" data-i18n-title="minimizeTitle">&times;</button>
    </div>
  </div>
  <div id="jobLogLink" class="hint"></div>
  <div id="log"></div>
  <div class="job-drawer-resize-handle" id="jobDrawerResizeHandle"></div>
</div>
<button type="button" class="drawer-reopen-pill" id="drawerReopenBtn">
  <span class="dot"></span><span id="drawerReopenText"></span>
</button>

<script>
let FORMATS = {}, EXCLUDES = {};
let MODELS = [];
let MODEL_ROOTS_COUNT = 0;
let LOGS_COUNT = 0;
let pollGeneration = 0;
let CURRENT_JOB_ID = null;
let CTQ_FOUND = false, CTQ_PATH = null;

function readStored(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch (e) { return fallback; }
}
function writeStored(key, value) {
  try { localStorage.setItem(key, value); } catch (e) {}
}

let LANG = readStored('lang', 'de');
let THEME = readStored('theme', 'dark');
let MINIMAL = readStored('minimal', '0') === '1';

const I18N = {
  de: {
    title: 'GFlava-Quant',
    brandSub: 'convert_to_quant \u00b7 laeuft nur lokal auf 127.0.0.1',
    introSummary: 'Kurz erklaert -- was macht Quantisierung?',
    introText: 'Quantisierung packt die Zahlen in den Modell-Gewichten in ein kompakteres Format ' +
      '(z.&nbsp;B. 8 statt 16&nbsp;Bit pro Zahl). Das macht die Datei kleiner und die Berechnung oft ' +
      'schneller, kostet aber ein kleines bisschen Bildqualitaet. Ein paar besonders empfindliche ' +
      'Layer (Ein-/Ausgabe-Projektionen, Normalisierungen) laesst man deshalb meist bewusst ' +
      'unangetastet -- welche das sind, legst du unten unter "Welche Layer ausschliessen" fest.',
    stepModelTitle: 'Modell &amp; Ausgabe',
    modelSelectLabel: 'Modell auswaehlen',
    modelSearchPh: 'Modell suchen (Name oder Ordner) ...',
    refreshModelsTitle: 'Ordner erneut einlesen',
    manualPathLabel: 'Pfad zur bf16-Modelldatei (.safetensors)',
    manualPathPh: 'z.B. C:\\ComfyUI\\models\\diffusion_models\\mein_modell.safetensors',
    manualPathHint: 'Muss eine bestehende .safetensors-Datei sein. Funktioniert auch mit ' +
      'eigenen gemergten/Community-Checkpoints, solange die Tensor-Namen zur Architektur ' +
      'passen, die du unten bei "Welche Layer ausschliessen" auswaehlst. Ueberschreibt die ' +
      'Dropdown-Auswahl oben.',
    outputPathLabel: 'Ausgabedatei (leer lassen fuer automatischen Namen im selben Ordner)',
    outputPathPh: 'wird automatisch vorgeschlagen',
    outputPathHint: 'Leer lassen erzeugt z.&nbsp;B. aus "modell.safetensors" automatisch ' +
      '"modell_int8_convrot.safetensors" im selben Ordner wie die Eingabedatei.',
    stepFormatTitle: 'Quantisierungsformat',
    groupsizeLabel: 'ConvRot-Gruppengroesse',
    groupsizeHint: 'Legt fest, wie viele Gewichts-Kanaele gemeinsam gedreht (Hadamard-Rotation) ' +
      'werden, bevor quantisiert wird -- das glaettet Ausreisser in den Zahlen und macht die ' +
      'Quantisierung genauer. Muss laut ctq-Dokumentation eine <b>Potenz von 4</b> sein: 4, 16, ' +
      '64, 256 oder 1024 (128 ist z.&nbsp;B. <u>ungueltig</u>, auch wenn es eine Zweierpotenz ' +
      'ist). <b>256 ist ctq\'s eigener Standardwert</b> und laut ctq-Dokumentation nicht ' +
      'modellspezifisch -- du kannst ihn fuer jede Architektur (Anima, Qwen-Image 2.1, Flux, ...) ' +
      'unveraendert lassen, ausser eine externe Konfiguration schreibt dir explizit einen anderen ' +
      'Wert vor.',
    blockSizeLabel: 'Block-Groesse (INT8 Block-Wise)',
    blockSizeHint: 'Groesse der Zahlenbloecke fuer die block-weise Skalierung. Laut ' +
      'ctq-Dokumentation ueblich: <b>64 oder 128</b> (Standard: 128) -- keine Potenz-von-4-Pflicht ' +
      'wie bei ConvRot, aber der Wert sollte die Layer-Dimensionen sinnvoll teilen. Ohne den ' +
      'dazugehoerigen Skalierungs-Modus wuerde ctq diesen Wert schlicht ignorieren und stattdessen ' +
      'tensor-weise quantisieren -- das Tool setzt <code>--scaling_mode block</code> hierfuer ' +
      'automatisch mit.',
    stepExcludeTitle: 'Welche Layer ausschliessen (unquantisiert lassen)',
    excludeIntro: 'Manche Layer reagieren besonders empfindlich auf Rundungsfehler -- ' +
      'typischerweise Ein-/Ausgabe-Projektionen, Zeitschritt-Einbettungen und ' +
      'Normalisierungs-Gewichte. Die bleiben deshalb in voller Genauigkeit (bf16), der Rest wird ' +
      'quantisiert. Ein Klick auf ein bekanntes Modell unten waehlt das passende Muster, ' +
      'oder waehle es direkt im Dropdown. Die Beschreibungen zu ctq\'s eigenen Presets sind woertlich ' +
      'aus <code>ctq --help-filters</code> deiner installierten Version uebernommen.',
    customRegexLabel: 'Eigenes --exclude-layers Regex',
    customRegexHint: 'Ein regulaerer Ausdruck, der gegen jeden Tensor-Namen geprueft wird ' +
      '(z.&nbsp;B. <code>transformer_blocks.0.attn.to_k.weight</code>). Trifft das Muster zu ' +
      '(an beliebiger Stelle im Namen), bleibt dieser Layer unquantisiert. Mehrere Begriffe mit ' +
      '<code>|</code> trennen: <code>(img_in|txt_in)</code> schliesst jeden Layer aus, dessen ' +
      'Name "img_in" ODER "txt_in" enthaelt.',
    advancedSummary: 'Erweitert (optionale ctq-Argumente)',
    lowMemoryLabel: 'Low-Memory-Modus (RAM-Streaming beim Laden)',
    lowMemoryAuto: 'Automatisch (empfohlen)',
    lowMemoryOn: 'Immer an',
    lowMemoryOff: 'Immer aus',
    lowMemoryHint: 'Steuert ctq\'s <code>--low-memory</code>-Flag. Laut ctq-Dokumentation und ' +
      'installiertem Quellcode betrifft das ausschliesslich den <b>System-RAM</b> beim Einlesen ' +
      '(alle Gewichte vorab in den RAM laden vs. einzeln von der Platte nachladen) -- ' +
      '<b>nicht den VRAM der Grafikkarte</b>: die GPU-Berechnung verschiebt bei ctq ohnehin immer ' +
      'nur einen Gewichts-Tensor auf einmal und gibt ihn danach wieder frei, unabhaengig von ' +
      'dieser Einstellung. "Automatisch" aktiviert es nur, wenn die Eingabedatei mehr als 50&nbsp;% ' +
      'des gerade verfuegbaren System-RAM belegt (ctq\'s eigene Empfehlung) -- kleinere Modelle ' +
      'werden dann schneller ohne Streaming-Overhead verarbeitet. Die tatsaechliche Entscheidung ' +
      'inkl. gemessener Werte steht im Log jedes Laufs.',
    extraArgsLabel: 'Zusaetzliche ctq-Argumente (frei, werden roh angehaengt)',
    extraArgsPh: 'z.B. --heur',
    extraArgsHint: 'Hier kannst du beliebige zusaetzliche ctq-Flags anhaengen, die diese ' +
      'Oberflaeche nicht extra abfragt -- freier Text, wird 1:1 an den Befehl angehaengt ' +
      '(mehrere Flags einfach mit Leerzeichen trennen). Ein paar Beispiele, die es laut ' +
      'ctq-Dokumentation tatsaechlich gibt: <code>--heur</code> (ueberspringt Layer mit fuer ' +
      'Quantisierung ungeeignetem Seitenverhaeltnis/Groesse), <code>--custom-layers REGEX ' +
      '--custom-type fp8</code> (bestimmte Layer mit einem anderen Format quantisieren als ' +
      'der Rest, hat Vorrang vor "Welche Layer ausschliessen"), <code>--fallback int8</code> ' +
      '(ausgeschlossene Layer statt in voller Praezision in einem anderen Quant-Format ' +
      'behalten). Die fuer <b>deine</b> installierte ctq-Version verbindliche, vollstaendige ' +
      'Liste zeigt dir der Button unten -- verlass dich im Zweifel lieber darauf als auf diese ' +
      'Beispiele hier.',
    ctqHelpShow: 'ctq --help anzeigen',
    ctqHelpHide: 'ctq --help ausblenden',
    ctqHelpLoading: 'Lade ...',
    logsTitle: 'Fruehere Laeufe',
    logsIntro: 'Das komplette Log jedes Quantisierungs-Laufs (voller ' +
      'ctq-Befehl + gesamte Ausgabe) wird dauerhaft unter <code>logs/</code> neben diesem Skript ' +
      'gespeichert -- bleibt also auch nach einem Server-Neustart erhalten und laesst sich hier ' +
      'jederzeit nachtraeglich pruefen.',
    runBtn: 'Quantisieren starten',
    runBtnRunning: 'Laeuft ...',
    minimizeTitle: 'Minimieren',
    ctqFound: 'ctq gefunden: ',
    ctqFoundShort: 'ctq gefunden',
    cancelJobBtn: 'Abbrechen',
    cancellingBtn: 'Wird abgebrochen ...',
    cancelConfirm: 'Laufende Quantisierung wirklich abbrechen?\n\nEine unvollstaendige Ausgabedatei wird geloescht (eine schon vorher vorhandene Datei bleibt unangetastet).',
    cancelledStatus: 'Abgebrochen.',
    cancelFailed: 'Abbrechen hat nicht geklappt.',
    minimalOnTitle: 'Minimales Design: keine Verlaeufe, eine Akzentfarbe',
    minimalOffTitle: 'Volles Design mit Verlaeufen',
    ctqMissingShort: 'ctq fehlt – hier installieren',
    ctqMissing: 'ctq wurde NICHT gefunden. Bitte zuerst "pip install convert-to-quant" im selben Python ausfuehren, dann diese Seite neu laden.',
    optgroupArch: 'Bekannte Architekturen (Community-Regex)',
    optgroupCtq: 'ctq eigene Presets (ctq -hf)',
    optgroupGeneral: 'Allgemein',
    formatUnverifiedSuffix: '  [nicht vollstaendig verifiziert]',
    modelSearching: 'Durchsuche Ordner ...',
    modelsFoundTemplate: (n, roots) => n + ' Modell(e) gefunden in ' + roots + ' durchsuchten Ordnern.',
    modelsNoneFound: 'Keine .safetensors-Dateien in den konfigurierten Ordnern gefunden -- Pfad manuell eingeben.',
    modelTypeDetecting: 'Erkenne Modelltyp ...',
    modelTypeDetectedTemplate: (label) => 'Erkannt: ' + label + ' -- Layer-Ausschluss automatisch gesetzt',
    modelsScanError: 'Fehler beim Einlesen der Modell-Ordner -- bitte Pfad manuell eingeben.',
    comboNoMatches: (q) => 'Keine Treffer fuer "' + q + '".',
    comboNoModels: 'Keine Modelle gefunden.',
    fullPathLabel: 'Voller Pfad: ',
    chooseModelFirst: 'Waehle zuerst ein Modell aus.',
    manualToggleShow: 'Pfad stattdessen manuell eingeben',
    manualToggleHide: 'Zurueck zur Modell-Liste',
    logsNone: 'Noch keine gespeicherten Logs -- erscheinen hier nach dem ersten Lauf.',
    logsError: 'Fehler beim Laden der Log-Liste.',
    logsView: 'ansehen',
    logsLoading: (name) => 'Lade ' + name + ' ...',
    logsLoadError: 'Fehler beim Laden des Logs.',
    statsBarTemplate: (roots, models, logs) => roots + ' Modell-Ordner durchsucht \u00b7 ' + models + ' Modelle gefunden \u00b7 ' + logs + ' gespeicherte Log(s)',
    badgeVerified: 'verifiziert',
    badgeExperimental: 'experimentell',
    formatVerifiedText: 'Dieses Format inkl. Marker-Fix wurde von uns Ende-zu-Ende in ComfyUI getestet.',
    formatUnverifiedText: 'Wird laut ctq-Dokumentation korrekt aufgerufen, aber das Marker-Format wurde von uns ' +
      'noch nicht gegen eine offizielle Datei verglichen -- Ergebnis vor Vertrauen einmal in ComfyUI pruefen.',
    badgeCommunityUnverified: 'Community, ungeprueft',
    excludeActivePattern: 'Aktives Muster: ',
    excludeUnquantizedNote: ' -- diese Namensbestandteile bleiben unquantisiert (bf16).',
    excludeCheckComfyUI: ' Vor dem Vertrauen einmal in ComfyUI pruefen.',
    badgeEndToEnd: 'Ende-zu-Ende getestet',
    excludeUsesBuiltin: 'Nutzt ctq\'s eingebautes Preset ',
    excludeFullDetails: ' Vollstaendige Details: "ctq --help anzeigen" unten bei Erweitert.',
    excludeNoneText: 'Es wird nichts ausgeschlossen -- normalerweise nicht empfohlen, ' +
      'weil dann auch empfindliche Layer (Normalisierungen, Ein-/Ausgabe) quantisiert werden.',
    excludeCustomText: 'Trag dein eigenes Muster im Feld darunter ein.',
    advancedBuiltinText: (value) => 'Fuer <code>' + value + '</code> sind laut ctq-Dokumentation keine ' +
      'zusaetzlichen Argumente noetig -- die Layer-Ausschluesse werden mit diesem Preset ' +
      'automatisch gesetzt (siehe Hinweis oben bei "Welche Layer ausschliessen"). Generell ' +
      'hilfreich und modellunabhaengig: <code>--heur</code> (ueberspringt Layer, deren ' +
      'Seitenverhaeltnis/Groesse sich schlecht fuer Quantisierung eignen).',
    advancedRegexText: 'Fuer dieses Community-Regex gibt es keine ctq-eigenen Zusatzhinweise, da ctq ' +
      'diese Architektur nicht nativ kennt. Bei Bedarf zusaetzlich <code>--heur</code> ergaenzen.',
    startingStatus: 'Starte ...',
    runningStatusTemplate: (pct) => 'Laeuft ... (' + pct + '%)',
    doneStatusTemplate: (path) => 'Fertig! Ausgabedatei: ' + path,
    errorStatusTemplate: (err) => 'Fehler: ' + err,
    durationLabel: 'Dauer: ',
    fullLogSavingAs: 'Vollstaendiges Log wird gespeichert als ',
    settingsTitle: 'Einstellungen',
    closeTitle: 'Schliessen',
    settingsHeading: 'Einstellungen',
    settingsIntro: 'Modell-Ordner, ComfyUI-Installation und ctq-Pfad sind hier hinterlegt statt ' +
      'fest im Code -- Aenderungen werden dauerhaft gespeichert und ueberleben einen Server-Neustart.',
    settingsModelRootsHeading: 'Modell-Ordner',
    settingsAddRoot: '+ Ordner hinzufuegen',
    settingsRemoveRoot: 'Ordner entfernen',
    settingsNoRoots: 'Noch kein Modell-Ordner konfiguriert -- "Ordner hinzufuegen" klicken.',
    settingsComfyuiHeading: 'ComfyUI-Installation',
    settingsComfyuiHint: 'Wird benoetigt, um custom_nodes zu finden (z.&nbsp;B. fuer den ' +
      'Systemcheck von ComfyUI-INT8-Fast unten).',
    settingsComfyuiPh: 'z.B. H:\\ComfyUI_windows_portable\\ComfyUI-Easy-Install\\ComfyUI',
    settingsCtqHeading: 'ctq-Programm (nur falls automatisch nicht gefunden)',
    settingsCtqPh: 'z.B. ...\\python_embeded\\Scripts\\ctq.exe',
    settingsCtqAutoFoundTemplate: (path) => 'Automatisch gefunden unter: ' + path + '. Nur eintragen, falls das nicht stimmt.',
    settingsCtqAutoMissing: 'Automatisch NICHT gefunden -- hier den Pfad zu ctq.exe eintragen oder ueber "Durchsuchen" suchen.',
    settingsCheckHeading: 'Systemcheck',
    settingsCheckHint: 'Prueft, ob ctq und das ComfyUI-INT8-Fast-Node installiert sind und ob ' +
      'dafuer Updates verfuegbar sind (braucht Internetzugriff).',
    settingsRunCheck: 'Jetzt pruefen',
    settingsChecking: 'Pruefe ...',
    settingsCheckFailed: 'Systemcheck fehlgeschlagen.',
    settingsInstall: 'Installieren',
    settingsUpdate: 'Aktualisieren',
    settingsWorking: 'Bitte warten ...',
    settingsActionFailed: 'Aktion fehlgeschlagen.',
    settingsCancel: 'Abbrechen',
    settingsSave: 'Speichern',
    settingsSaveError: 'Speichern fehlgeschlagen.',
    settingsBrowse: 'Durchsuchen',
    folderPickerHint: 'Diesen Ordner waehlen (Oeffnen klicken)',
    firstRunBanner: 'Willkommen! Bitte richte unter Einstellungen mindestens einen Modell-Ordner ein.',
  },
  en: {
    title: 'GFlava-Quant',
    brandSub: 'convert_to_quant \u00b7 runs locally on 127.0.0.1 only',
    introSummary: 'Quick overview -- what does quantization do?',
    introText: 'Quantization packs the numbers in the model weights into a more compact format ' +
      '(e.g.&nbsp;8 instead of 16&nbsp;bits per number). This makes the file smaller and computation ' +
      'often faster, at the cost of a small amount of image quality. A few particularly sensitive ' +
      'layers (input/output projections, normalizations) are therefore usually left untouched on ' +
      'purpose -- which ones exactly, you set below under "Which layers to exclude".',
    stepModelTitle: 'Model &amp; Output',
    modelSelectLabel: 'Select model',
    modelSearchPh: 'Search model (name or folder) ...',
    refreshModelsTitle: 'Rescan folders',
    manualPathLabel: 'Path to the bf16 model file (.safetensors)',
    manualPathPh: 'e.g. C:\\ComfyUI\\models\\diffusion_models\\my_model.safetensors',
    manualPathHint: 'Must be an existing .safetensors file. Also works with your own ' +
      'merged/community checkpoints, as long as the tensor names match the architecture you ' +
      'select below under "Which layers to exclude". Overrides the dropdown selection above.',
    outputPathLabel: 'Output file (leave empty for an automatic name in the same folder)',
    outputPathPh: 'will be suggested automatically',
    outputPathHint: 'Leaving it empty generates e.g.&nbsp;"model_int8_convrot.safetensors" from ' +
      '"model.safetensors" automatically, in the same folder as the input file.',
    stepFormatTitle: 'Quantization format',
    groupsizeLabel: 'ConvRot group size',
    groupsizeHint: 'Determines how many weight channels are rotated together (Hadamard rotation) ' +
      'before quantizing -- this smooths outliers in the numbers and makes quantization more ' +
      'accurate. Must be a <b>power of 4</b> per ctq\'s documentation: 4, 16, 64, 256 or 1024 ' +
      '(128 is <u>invalid</u>, for example, even though it is a power of 2). <b>256 is ctq\'s own ' +
      'default</b> and, per ctq\'s documentation, not model-specific -- you can leave it unchanged ' +
      'for any architecture (Anima, Qwen-Image 2.1, Flux, ...) unless an external configuration ' +
      'explicitly prescribes a different value.',
    blockSizeLabel: 'Block size (INT8 Block-Wise)',
    blockSizeHint: 'Size of the number blocks for block-wise scaling. Common per ctq\'s ' +
      'documentation: <b>64 or 128</b> (default: 128) -- no power-of-4 requirement like ConvRot, ' +
      'but the value should evenly divide the layer dimensions. Without the matching scaling ' +
      'mode, ctq would simply ignore this value and quantize tensor-wise instead -- this tool ' +
      'automatically adds <code>--scaling_mode block</code> for it.',
    stepExcludeTitle: 'Which layers to exclude (keep unquantized)',
    excludeIntro: 'Some layers react especially sensitively to rounding errors -- typically ' +
      'input/output projections, timestep embeddings, and normalization weights. These are ' +
      'therefore kept at full precision (bf16), the rest gets quantized. Clicking a known model ' +
      'below picks the matching pattern, or pick one directly in the dropdown. The descriptions ' +
      'of ctq\'s own presets are taken verbatim from <code>ctq --help-filters</code> of your ' +
      'installed version.',
    customRegexLabel: 'Custom --exclude-layers regex',
    customRegexHint: 'A regular expression that is checked against every tensor name (e.g.&nbsp;' +
      '<code>transformer_blocks.0.attn.to_k.weight</code>). If the pattern matches (anywhere in ' +
      'the name), that layer stays unquantized. Separate multiple terms with <code>|</code>: ' +
      '<code>(img_in|txt_in)</code> excludes any layer whose name contains "img_in" OR "txt_in".',
    advancedSummary: 'Advanced (optional ctq arguments)',
    lowMemoryLabel: 'Low-memory mode (RAM streaming while loading)',
    lowMemoryAuto: 'Automatic (recommended)',
    lowMemoryOn: 'Always on',
    lowMemoryOff: 'Always off',
    lowMemoryHint: 'Controls ctq\'s <code>--low-memory</code> flag. Per ctq\'s own documentation and ' +
      'installed source code, this affects only <b>system RAM</b> while loading (preloading all ' +
      'weights into RAM upfront vs. reading them from disk one at a time) -- ' +
      '<b>not the GPU\'s VRAM</b>: ctq\'s GPU processing always moves only one weight tensor at a ' +
      'time and frees it right after, regardless of this setting. "Automatic" only enables it when ' +
      'the input file uses more than 50% of the currently available system RAM (ctq\'s own ' +
      'recommendation) -- smaller models are then processed faster without streaming overhead. The ' +
      'actual decision, including the measured values, is written to each run\'s log.',
    extraArgsLabel: 'Additional ctq arguments (free text, appended raw)',
    extraArgsPh: 'e.g. --heur',
    extraArgsHint: 'Here you can append any additional ctq flags that this interface doesn\'t ' +
      'ask for separately -- free text, appended 1:1 to the command (separate multiple flags ' +
      'with spaces). A few examples that actually exist per ctq\'s documentation: ' +
      '<code>--heur</code> (skips layers with an aspect ratio/size unsuitable for quantization), ' +
      '<code>--custom-layers REGEX --custom-type fp8</code> (quantize certain layers with a ' +
      'different format than the rest, takes priority over "Which layers to exclude"), ' +
      '<code>--fallback int8</code> (keep excluded layers in a different quant format instead of ' +
      'full precision). The complete list binding for <b>your</b> installed ctq version is shown ' +
      'by the button below -- when in doubt, rely on that rather than these examples.',
    ctqHelpShow: 'Show ctq --help',
    ctqHelpHide: 'Hide ctq --help',
    ctqHelpLoading: 'Loading ...',
    logsTitle: 'Previous runs',
    logsIntro: 'The complete log of every quantization run (full ctq command + entire output) is ' +
      'permanently saved under <code>logs/</code> next to this script -- so it survives a server ' +
      'restart and can be reviewed here at any time afterwards.',
    runBtn: 'Start quantization',
    runBtnRunning: 'Running ...',
    minimizeTitle: 'Minimize',
    ctqFound: 'ctq found: ',
    ctqFoundShort: 'ctq found',
    cancelJobBtn: 'Cancel',
    cancellingBtn: 'Cancelling ...',
    cancelConfirm: 'Really cancel the running quantization?\n\nAn incomplete output file will be deleted (a file that already existed before stays untouched).',
    cancelledStatus: 'Cancelled.',
    cancelFailed: 'Cancelling failed.',
    minimalOnTitle: 'Minimal design: no gradients, one accent color',
    minimalOffTitle: 'Full design with gradients',
    ctqMissingShort: 'ctq missing – install here',
    ctqMissing: 'ctq was NOT found. Please run "pip install convert-to-quant" in the same Python first, then reload this page.',
    optgroupArch: 'Known architectures (community regex)',
    optgroupCtq: 'ctq\'s own presets (ctq -hf)',
    optgroupGeneral: 'General',
    formatUnverifiedSuffix: '  [not fully verified]',
    modelSearching: 'Scanning folders ...',
    modelsFoundTemplate: (n, roots) => n + ' model(s) found in ' + roots + ' scanned folders.',
    modelsNoneFound: 'No .safetensors files found in the configured folders -- enter the path manually.',
    modelTypeDetecting: 'Detecting model type ...',
    modelTypeDetectedTemplate: (label) => 'Detected: ' + label + ' -- layer exclusion set automatically',
    modelsScanError: 'Error scanning the model folders -- please enter the path manually.',
    comboNoMatches: (q) => 'No matches for "' + q + '".',
    comboNoModels: 'No models found.',
    fullPathLabel: 'Full path: ',
    chooseModelFirst: 'Select a model first.',
    manualToggleShow: 'Enter path manually instead',
    manualToggleHide: 'Back to model list',
    logsNone: 'No saved logs yet -- they will appear here after the first run.',
    logsError: 'Error loading the log list.',
    logsView: 'view',
    logsLoading: (name) => 'Loading ' + name + ' ...',
    logsLoadError: 'Error loading the log.',
    statsBarTemplate: (roots, models, logs) => roots + ' model folders scanned \u00b7 ' + models + ' models found \u00b7 ' + logs + ' saved log(s)',
    badgeVerified: 'verified',
    badgeExperimental: 'experimental',
    formatVerifiedText: 'This format including the marker fix was tested end-to-end by us in ComfyUI.',
    formatUnverifiedText: 'Called correctly per ctq\'s documentation, but we have not compared the marker ' +
      'format against an official file yet -- check the result once in ComfyUI before relying on it.',
    badgeCommunityUnverified: 'community, unverified',
    excludeActivePattern: 'Active pattern: ',
    excludeUnquantizedNote: ' -- these name fragments stay unquantized (bf16).',
    excludeCheckComfyUI: ' Check it once in ComfyUI before relying on it.',
    badgeEndToEnd: 'end-to-end tested',
    excludeUsesBuiltin: 'Uses ctq\'s built-in preset ',
    excludeFullDetails: ' Full details: "Show ctq --help" below under Advanced.',
    excludeNoneText: 'Nothing gets excluded -- usually not recommended, since sensitive layers ' +
      '(normalizations, input/output) would get quantized too.',
    excludeCustomText: 'Enter your own pattern in the field below.',
    advancedBuiltinText: (value) => 'For <code>' + value + '</code>, no additional arguments are needed ' +
      'per ctq\'s documentation -- the layer exclusions are set automatically by this preset (see the ' +
      'hint above under "Which layers to exclude"). Generally helpful and model-independent: ' +
      '<code>--heur</code> (skips layers whose aspect ratio/size are poorly suited for quantization).',
    advancedRegexText: 'There are no ctq-specific extra hints for this community regex, since ctq ' +
      'does not know this architecture natively. Add <code>--heur</code> additionally if needed.',
    startingStatus: 'Starting ...',
    runningStatusTemplate: (pct) => 'Running ... (' + pct + '%)',
    doneStatusTemplate: (path) => 'Done! Output file: ' + path,
    errorStatusTemplate: (err) => 'Error: ' + err,
    durationLabel: 'Duration: ',
    fullLogSavingAs: 'Full log is being saved as ',
    settingsTitle: 'Settings',
    closeTitle: 'Close',
    settingsHeading: 'Settings',
    settingsIntro: 'Model folders, ComfyUI installation and the ctq path are stored here instead ' +
      'of hardcoded in the code -- changes are saved permanently and survive a server restart.',
    settingsModelRootsHeading: 'Model folders',
    settingsAddRoot: '+ Add folder',
    settingsRemoveRoot: 'Remove folder',
    settingsNoRoots: 'No model folder configured yet -- click "Add folder".',
    settingsComfyuiHeading: 'ComfyUI installation',
    settingsComfyuiHint: 'Needed to locate custom_nodes (e.g. for the ComfyUI-INT8-Fast system ' +
      'check below).',
    settingsComfyuiPh: 'e.g. H:\\ComfyUI_windows_portable\\ComfyUI-Easy-Install\\ComfyUI',
    settingsCtqHeading: 'ctq program (only if not found automatically)',
    settingsCtqPh: 'e.g. ...\\python_embeded\\Scripts\\ctq.exe',
    settingsCtqAutoFoundTemplate: (path) => 'Automatically found at: ' + path + '. Only set this if that is wrong.',
    settingsCtqAutoMissing: 'NOT found automatically -- enter the path to ctq.exe here or use "Browse".',
    settingsCheckHeading: 'System check',
    settingsCheckHint: 'Checks whether ctq and the ComfyUI-INT8-Fast node are installed and whether ' +
      'updates are available for them (needs internet access).',
    settingsRunCheck: 'Check now',
    settingsChecking: 'Checking ...',
    settingsCheckFailed: 'System check failed.',
    settingsInstall: 'Install',
    settingsUpdate: 'Update',
    settingsWorking: 'Please wait ...',
    settingsActionFailed: 'Action failed.',
    settingsCancel: 'Cancel',
    settingsSave: 'Save',
    settingsSaveError: 'Saving failed.',
    settingsBrowse: 'Browse',
    folderPickerHint: 'Select this folder (click Open)',
    firstRunBanner: 'Welcome! Please set up at least one model folder under Settings.',
  },
};

function t(key, ...args) {
  const dict = I18N[LANG] || I18N.de;
  const val = (dict[key] !== undefined) ? dict[key] : I18N.de[key];
  return typeof val === 'function' ? val(...args) : val;
}

function applyStaticI18n() {
  document.title = t('title');
  document.documentElement.lang = LANG;
  document.querySelectorAll('[data-i18n]').forEach((el) => { el.innerHTML = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-ph]').forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
  document.querySelectorAll('[data-i18n-title]').forEach((el) => {
    const txt = t(el.dataset.i18nTitle);
    el.title = txt;
    el.setAttribute('aria-label', txt);
  });
}

function renderCtqStatus() {
  const ctqEl = document.getElementById('ctqStatus');
  const ctqText = document.getElementById('ctqStatusText');
  ctqEl.classList.remove('ok', 'missing');
  if (CTQ_FOUND) {
    ctqText.textContent = t('ctqFoundShort');
    ctqEl.title = t('ctqFound') + CTQ_PATH;
    ctqEl.classList.add('ok');
  } else {
    ctqText.textContent = t('ctqMissingShort');
    ctqEl.title = t('ctqMissing');
    ctqEl.classList.add('missing');
  }
}

function formatLabelFor(entry) {
  return LANG === 'en' ? (entry.label_en || entry.label) : entry.label;
}

function populateFormatSelect() {
  const fmtSel = document.getElementById('formatSelect');
  const prev = fmtSel.value || 'int8_convrot';
  fmtSel.innerHTML = '';
  for (const [key, val] of Object.entries(FORMATS)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = formatLabelFor(val) + (val.verified ? '' : t('formatUnverifiedSuffix'));
    fmtSel.appendChild(opt);
  }
  fmtSel.value = prev;
}

function populateExcludeSelect() {
  const exSel = document.getElementById('excludeSelect');
  const prev = exSel.value || 'qwen21_custom';
  exSel.innerHTML = '';
  const grpArch = document.createElement('optgroup'); grpArch.label = t('optgroupArch');
  const grpCtq = document.createElement('optgroup'); grpCtq.label = t('optgroupCtq');
  const grpGeneral = document.createElement('optgroup'); grpGeneral.label = t('optgroupGeneral');
  for (const [key, val] of Object.entries(EXCLUDES)) {
    const opt = document.createElement('option');
    opt.value = key; opt.textContent = formatLabelFor(val);
    if (val.kind === 'builtin') grpCtq.appendChild(opt);
    else if (val.kind === 'regex') grpArch.appendChild(opt);
    else grpGeneral.appendChild(opt);
  }
  exSel.appendChild(grpArch); exSel.appendChild(grpCtq); exSel.appendChild(grpGeneral);
  exSel.value = prev;
}

function applyTheme() {
  if (THEME === 'light') document.documentElement.setAttribute('data-theme', 'light');
  else document.documentElement.removeAttribute('data-theme');
  drawBackground();
}

const BG_PALETTES = {
  dark: {
    base: [7, 8, 13], stops: [[92, 30, 130], [32, 44, 122], [14, 100, 142]], strength: 0.62,
    dot: 'rgba(255,255,255,0.045)',
  },
  light: {
    base: [238, 240, 246], stops: [[196, 160, 228], [166, 180, 234], [150, 210, 232]], strength: 0.5,
    dot: 'rgba(20,24,40,0.06)',
  },
};

// Dreiecks-Mosaik = quantisierter Farbverlauf: Position und Helligkeit werden
// auf wenige feste Stufen gerundet. Dazu das Punktraster der ComfyUI-Arbeitsflaeche.
function drawBackground() {
  const cv = document.getElementById('bgCanvas');
  if (!cv) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  const w = window.innerWidth, h = window.innerHeight;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pal = document.documentElement.getAttribute('data-theme') === 'light' ? BG_PALETTES.light : BG_PALETTES.dark;
  const lerp = (a, b, t) => a + (b - a) * t;
  const hash = (a, b) => { const s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return s - Math.floor(s); };
  const rgb = (c) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

  ctx.fillStyle = rgb(pal.base);
  ctx.fillRect(0, 0, w, h);
  const drawDots = () => {
    ctx.fillStyle = pal.dot;
    for (let y = 12; y < h; y += 24) for (let x = 12; x < w; x += 24) ctx.fillRect(x, y, 1.5, 1.5);
  };
  if (document.documentElement.hasAttribute('data-minimal')) { drawDots(); return; }

  const side = 64, th = side * Math.sqrt(3) / 2;
  const HUE_STEPS = 7, LIGHT_STEPS = 4;
  for (let r = -1; r * th < h + th; r++) {
    for (let c = -2; c * side / 2 < w + side; c++) {
      const x = c * side / 2, y = r * th;
      const up = ((c + r) % 2 + 2) % 2 === 0;
      const cx = x + side / 2, cy = y + th / 2;
      const t = Math.round(Math.min(1, Math.max(0, (cx / w) * 0.55 + (1 - cy / h) * 0.45)) * HUE_STEPS) / HUE_STEPS;
      const col = t < 0.5
        ? pal.stops[0].map((v, i) => lerp(v, pal.stops[1][i], t * 2))
        : pal.stops[1].map((v, i) => lerp(v, pal.stops[2][i], (t - 0.5) * 2));
      const level = Math.round(hash(r, c) * LIGHT_STEPS) / LIGHT_STEPS;
      const vignette = 1 - 0.45 * Math.pow(Math.min(1, Math.max(0, cy / h)), 3);
      const k = pal.strength * (0.42 + 0.4 * level) * vignette;
      const fill = rgb(pal.base.map((v, i) => lerp(v, col[i], k)));
      ctx.beginPath();
      if (up) { ctx.moveTo(x, y + th); ctx.lineTo(x + side / 2, y); ctx.lineTo(x + side, y + th); }
      else { ctx.moveTo(x, y); ctx.lineTo(x + side, y); ctx.lineTo(x + side / 2, y + th); }
      ctx.closePath();
      ctx.fillStyle = fill; ctx.strokeStyle = fill; ctx.lineWidth = 1;
      ctx.fill(); ctx.stroke();
    }
  }

  drawDots();
}

let bgResizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(bgResizeTimer);
  bgResizeTimer = setTimeout(drawBackground, 150);
});

function toggleTheme() {
  THEME = (THEME === 'light') ? 'dark' : 'light';
  writeStored('theme', THEME);
  // Ohne das starten beim Umschalten ~300 Farb-Transitions gleichzeitig.
  const root = document.documentElement;
  root.classList.add('no-transitions');
  applyTheme();
  requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove('no-transitions')));
}

function updateMinimalButton() {
  const btn = document.getElementById('minimalToggleBtn');
  const label = MINIMAL ? t('minimalOffTitle') : t('minimalOnTitle');
  btn.title = label;
  btn.setAttribute('aria-label', label);
  btn.setAttribute('aria-pressed', MINIMAL ? 'true' : 'false');
}

function applyMinimal() {
  if (MINIMAL) document.documentElement.setAttribute('data-minimal', '');
  else document.documentElement.removeAttribute('data-minimal');
  updateMinimalButton();
  drawBackground();
}

function toggleMinimal() {
  MINIMAL = !MINIMAL;
  writeStored('minimal', MINIMAL ? '1' : '0');
  const root = document.documentElement;
  root.classList.add('no-transitions');
  applyMinimal();
  requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove('no-transitions')));
}

function applyLanguage() {
  applyStaticI18n();
  renderCtqStatus();
  updateMinimalButton();
  populateFormatSelect();
  populateExcludeSelect();
  updateFormatHint();
  updateExcludeHint();
  updateStatsBar();
  if (selectedModel) { selectModel(selectedModel); } else { updateActionSummary(); }
  showManualPath(document.getElementById('manualPathRow').style.display !== 'none');
  document.getElementById('modelSelectHint').textContent = MODELS.length
    ? t('modelsFoundTemplate', MODELS.length, MODEL_ROOTS_COUNT)
    : t('modelsNoneFound');
  if (!document.getElementById('runBtn').disabled) {
    document.getElementById('runBtn').textContent = t('runBtn');
  }
  loadLogsList();
  if (document.getElementById('settingsOverlay').classList.contains('open')) {
    renderRootList();
    updateCtqAutoHint();
    if (LAST_SYSTEM_CHECKS) renderSystemCheck(LAST_SYSTEM_CHECKS);
  }
}

function setLanguage(lang) {
  LANG = lang;
  writeStored('lang', LANG);
  applyLanguage();
}

// ---------------------------------------------------------------------------
// Einstellungen: Modell-Ordner, ComfyUI-Pfad, ctq-Override, Systemcheck.
// ---------------------------------------------------------------------------

let SETTINGS = { model_roots: [], comfyui_root: '', ctq_path_override: '' };
let CTQ_AUTO_PATH = null;
let LAST_SYSTEM_CHECKS = null;

function updateCtqAutoHint() {
  const el = document.getElementById('settingsCtqAutoHint');
  el.textContent = CTQ_AUTO_PATH ? t('settingsCtqAutoFoundTemplate', CTQ_AUTO_PATH) : t('settingsCtqAutoMissing');
}

function renderRootList() {
  const list = document.getElementById('settingsRootList');
  if (!SETTINGS.model_roots.length) {
    list.innerHTML = `<div class="modal-hint" style="margin:0 0 10px;">${t('settingsNoRoots')}</div>`;
    return;
  }
  list.innerHTML = SETTINGS.model_roots.map((r, i) => `
    <div class="root-row">
      <svg class="root-row-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
      <div class="root-row-text">
        <div class="root-row-label">${escapeHtml(r.label)}</div>
        <div class="root-row-path">${escapeHtml(r.path)}</div>
      </div>
      <button type="button" class="secondary icon-btn" data-remove-idx="${i}" title="${t('settingsRemoveRoot')}" aria-label="${t('settingsRemoveRoot')}">&times;</button>
    </div>
  `).join('');
  list.querySelectorAll('[data-remove-idx]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      SETTINGS.model_roots.splice(+e.currentTarget.dataset.removeIdx, 1);
      renderRootList();
    });
  });
}

function addModelRootFromPath(path) {
  const name = path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || path;
  SETTINGS.model_roots.push({ key: '', label: name, path });
  renderRootList();
}

async function openSettings(showFirstRunHint) {
  const errEl = document.getElementById('settingsError');
  errEl.style.display = 'none';
  document.getElementById('systemCheckList').innerHTML = '';
  LAST_SYSTEM_CHECKS = null;
  try {
    const res = await fetch('/api/settings');
    const data = await res.json();
    SETTINGS = {
      model_roots: (data.model_roots || []).map((r) => ({ ...r })),
      comfyui_root: data.comfyui_root || '',
      ctq_path_override: data.ctq_path_override || '',
    };
    CTQ_AUTO_PATH = data.ctq_auto_path || null;
  } catch (e) {
    SETTINGS = { model_roots: [], comfyui_root: '', ctq_path_override: '' };
    CTQ_AUTO_PATH = null;
  }
  document.getElementById('comfyuiRootInput').value = SETTINGS.comfyui_root;
  document.getElementById('ctqOverrideInput').value = SETTINGS.ctq_path_override;
  updateCtqAutoHint();
  renderRootList();
  document.getElementById('settingsOverlay').classList.add('open');
  document.querySelector('.topbar').classList.add('settings-open');
  document.getElementById('settingsBtn').setAttribute('aria-expanded', 'true');
  if (showFirstRunHint) {
    errEl.style.color = 'var(--accent-bright)';
    errEl.textContent = t('firstRunBanner');
    errEl.style.display = 'block';
  } else {
    errEl.style.color = 'var(--error)';
  }
}

function closeSettings() {
  document.getElementById('settingsOverlay').classList.remove('open');
  document.querySelector('.topbar').classList.remove('settings-open');
  document.getElementById('settingsBtn').setAttribute('aria-expanded', 'false');
}

function settingsIsOpen() {
  return document.getElementById('settingsOverlay').classList.contains('open');
}

async function saveSettings() {
  const errEl = document.getElementById('settingsError');
  errEl.style.color = 'var(--error)';
  errEl.style.display = 'none';
  const payload = {
    model_roots: SETTINGS.model_roots,
    comfyui_root: document.getElementById('comfyuiRootInput').value.trim(),
    ctq_path_override: document.getElementById('ctqOverrideInput').value.trim(),
  };
  try {
    const res = await fetch('/api/settings', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      errEl.textContent = data.error || t('settingsSaveError');
      errEl.style.display = 'block';
      return;
    }
    closeSettings();
    await loadConfig();
    await loadModels();
  } catch (e) {
    errEl.textContent = t('settingsSaveError');
    errEl.style.display = 'block';
  }
}

const CHECK_ICON_SVG = {
  ok: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
  warn: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
  missing: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
  unknown: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.5-2 1.8-2 3.4"></path><line x1="12" y1="16.5" x2="12.01" y2="16.5"></line></svg>',
};

function renderSystemCheck(checks) {
  LAST_SYSTEM_CHECKS = checks;
  const listEl = document.getElementById('systemCheckList');
  listEl.innerHTML = checks.map((c) => {
    const label = formatLabelFor(c);
    const detail = LANG === 'en' ? (c.detail_en || c.detail) : c.detail;
    const actions = [];
    if (c.can_install) actions.push(`<button type="button" data-check-action="install" data-check-id="${c.id}">${t('settingsInstall')}</button>`);
    if (c.can_update) actions.push(`<button type="button" class="secondary" data-check-action="update" data-check-id="${c.id}">${t('settingsUpdate')}</button>`);
    return `
      <div class="check-row">
        <div class="check-icon ${c.status}">${CHECK_ICON_SVG[c.status] || CHECK_ICON_SVG.unknown}</div>
        <div class="check-row-text">
          <div class="check-row-label">${escapeHtml(label)}</div>
          <div class="check-row-detail">${escapeHtml(detail)}</div>
        </div>
        <div class="check-row-actions">${actions.join('')}</div>
      </div>`;
  }).join('');
  listEl.querySelectorAll('[data-check-action]').forEach((btn) => {
    btn.addEventListener('click', () => performCheckAction(btn.dataset.checkId, btn.dataset.checkAction, btn));
  });
}

async function runSystemCheck() {
  const listEl = document.getElementById('systemCheckList');
  listEl.innerHTML = `<div class="modal-hint" style="margin:0 0 10px;">${t('settingsChecking')}</div>`;
  try {
    const res = await fetch('/api/system_check');
    const data = await res.json();
    renderSystemCheck(data.checks || []);
  } catch (e) {
    listEl.innerHTML = `<div class="modal-hint" style="margin:0 0 10px; color: var(--error);">${t('settingsCheckFailed')}</div>`;
  }
}

async function performCheckAction(id, action, btn) {
  const errEl = document.getElementById('settingsError');
  errEl.style.color = 'var(--error)';
  errEl.style.display = 'none';
  btn.disabled = true;
  btn.textContent = t('settingsWorking');
  try {
    const res = await fetch('/api/system_check/action', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id, action }),
    });
    const data = await res.json();
    if (!res.ok) {
      errEl.textContent = data.error || t('settingsActionFailed');
      errEl.style.display = 'block';
    }
  } catch (e) {
    errEl.textContent = t('settingsActionFailed');
    errEl.style.display = 'block';
  }
  await runSystemCheck();
  if (id === 'ctq') await loadConfig();
}

// Echte native Windows-Dialoge (FolderBrowserDialog/OpenFileDialog) statt
// eines selbstgebauten In-App-Ordner-Browsers -- der Server oeffnet sie ueber
// eingebettetes PowerShell (siehe /api/browse_native), genau wie
// start_server.bat es fuer die python.exe-Auswahl macht. Der Aufruf
// blockiert, bis der Nutzer waehlt oder abbricht.
async function pickNativeFolder(title, initialDir) {
  const errEl = document.getElementById('settingsError');
  try {
    const res = await fetch('/api/browse_native', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'folder', title, initial_dir: initialDir || '', select_hint: t('folderPickerHint') }),
    });
    const data = await res.json();
    if (!res.ok) {
      errEl.style.color = 'var(--error)';
      errEl.textContent = data.error || t('settingsActionFailed');
      errEl.style.display = 'block';
      return null;
    }
    return data.path || null;
  } catch (e) {
    errEl.style.color = 'var(--error)';
    errEl.textContent = t('settingsActionFailed');
    errEl.style.display = 'block';
    return null;
  }
}

async function pickNativeFile(title, filter, initialDir) {
  const errEl = document.getElementById('settingsError');
  try {
    const res = await fetch('/api/browse_native', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'file', title, filter, initial_dir: initialDir || '' }),
    });
    const data = await res.json();
    if (!res.ok) {
      errEl.style.color = 'var(--error)';
      errEl.textContent = data.error || t('settingsActionFailed');
      errEl.style.display = 'block';
      return null;
    }
    return data.path || null;
  } catch (e) {
    errEl.style.color = 'var(--error)';
    errEl.textContent = t('settingsActionFailed');
    errEl.style.display = 'block';
    return null;
  }
}

// Alle echten Architektur-Presets (alles aus EXCLUDE_PRESETS ausser den
// Pseudo-Eintraegen "none"/"custom") -- jedes Preset, das ctq per
// `--help-filters` kennt, plus unsere drei Community-Regexes, damit hier
// wirklich jeder Modelltyp per Klick erreichbar ist, den ctq bearbeiten kann.
const CHIP_KEYS = [
  // Community-Regex (nicht von ctq selbst, aber haeufig gebraucht)
  'qwen21_custom', 'flux1_custom', 'sdxl_custom',
  // ctq --help-filters: Image Models
  'ctq_qwen', 'ctq_zimage', 'ctq_zimage_refiner', 'ctq_boogu',
  // ctq --help-filters: Diffusion Models (Flux-style)
  'ctq_anima', 'ctq_lens', 'ctq_flux2', 'ctq_distillation_large', 'ctq_distillation_small',
  'ctq_nerf_large', 'ctq_nerf_small', 'ctq_radiance', 'ctq_krea2', 'ctq_ideogram4',
  // ctq --help-filters: Video Models
  'ctq_wan', 'ctq_hunyuan', 'ctq_minimaxh3', 'ctq_ltxv2',
  // ctq --help-filters: Text Encoders
  'ctq_gemma4', 'ctq_qwen_vlm', 'ctq_t5xxl', 'ctq_mistral', 'ctq_visual', 'ctq_generic_text',
];

async function loadConfig() {
  const res = await fetch('/api/config');
  const cfg = await res.json();
  FORMATS = cfg.formats; EXCLUDES = cfg.excludes;
  CTQ_FOUND = cfg.ctq_found; CTQ_PATH = cfg.ctq_path;
  renderCtqStatus();

  populateFormatSelect();
  document.getElementById('formatSelect').value = 'int8_convrot';
  updateFormatHint();
  document.getElementById('formatSelect').addEventListener('change', updateFormatHint);

  populateExcludeSelect();
  document.getElementById('excludeSelect').value = 'qwen21_custom';
  document.getElementById('excludeSelect').addEventListener('change', () => {
    const exSel = document.getElementById('excludeSelect');
    document.getElementById('customRegexRow').style.display = (exSel.value === 'custom') ? 'block' : 'none';
    updateExcludeHint();
    updateChipActive();
  });
  renderChips();
  updateExcludeHint();
}

function applyExcludePreset(key) {
  const exSel = document.getElementById('excludeSelect');
  exSel.value = key;
  document.getElementById('customRegexRow').style.display = (key === 'custom') ? 'block' : 'none';
  updateExcludeHint();
  updateChipActive();
}

function renderChips() {
  const box = document.getElementById('presetChips');
  box.innerHTML = '';
  for (const key of CHIP_KEYS) {
    const val = EXCLUDES[key];
    if (!val) continue;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chip';
    btn.dataset.key = key;
    btn.textContent = val.chip || val.label;
    btn.addEventListener('click', () => applyExcludePreset(key));
    box.appendChild(btn);
  }
  updateChipActive();
}

function updateChipActive() {
  const cur = document.getElementById('excludeSelect').value;
  document.querySelectorAll('#presetChips .chip').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.key === cur);
  });
}

let selectedModel = null;
let comboVisibleItems = [];
let comboActiveIndex = -1;

async function loadModels() {
  const hint = document.getElementById('modelSelectHint');
  hint.textContent = t('modelSearching');
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    MODELS = data.models || [];
    MODEL_ROOTS_COUNT = (data.roots || []).length;
    if (selectedModel) {
      const stillThere = MODELS.find((m) => m.path === selectedModel.path);
      if (stillThere) {
        selectModel(stillThere);
      } else {
        selectedModel = null;
        document.getElementById('modelSearch').value = '';
        document.getElementById('modelPathPreview').style.display = 'none';
        document.getElementById('modelTypePill').style.display = 'none';
        updateActionSummary();
      }
    }
    hint.textContent = MODELS.length
      ? t('modelsFoundTemplate', MODELS.length, MODEL_ROOTS_COUNT)
      : t('modelsNoneFound');
  } catch (e) {
    hint.textContent = t('modelsScanError');
    showManualPath(true);
  }
  updateStatsBar();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function modelDisplayName(m) {
  return (m.subdir ? m.subdir + '/' : '') + m.filename;
}

function modelMatches(m, q) {
  if (!q) return true;
  const hay = (modelDisplayName(m) + ' ' + m.root_label).toLowerCase();
  return hay.includes(q.toLowerCase());
}

function highlightMatch(text, query) {
  if (!query) return escapeHtml(text);
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return escapeHtml(text);
  return escapeHtml(text.slice(0, idx)) + '<mark>' + escapeHtml(text.slice(idx, idx + query.length))
    + '</mark>' + escapeHtml(text.slice(idx + query.length));
}

function renderComboPanel(query) {
  const panel = document.getElementById('modelComboPanel');
  panel.innerHTML = '';
  const filtered = MODELS.filter((m) => modelMatches(m, query));
  comboVisibleItems = filtered;
  comboActiveIndex = filtered.length ? 0 : -1;

  if (!filtered.length) {
    const empty = document.createElement('div');
    empty.className = 'combo-empty';
    empty.textContent = MODELS.length ? t('comboNoMatches', query) : t('comboNoModels');
    panel.appendChild(empty);
    return;
  }

  let lastGroup = null;
  filtered.forEach((m, idx) => {
    if (m.root_label !== lastGroup) {
      lastGroup = m.root_label;
      const label = document.createElement('div');
      label.className = 'combo-group-label';
      label.textContent = m.root_label;
      panel.appendChild(label);
    }
    const item = document.createElement('div');
    item.className = 'combo-item';
    item.dataset.idx = idx;
    const name = document.createElement('span');
    name.innerHTML = highlightMatch(modelDisplayName(m), query);
    const size = document.createElement('span');
    size.className = 'combo-item-size';
    size.textContent = m.size_mb != null ? m.size_mb + ' MB' : '';
    item.appendChild(name); item.appendChild(size);
    item.addEventListener('mousedown', (e) => { e.preventDefault(); selectModel(m); });
    item.addEventListener('mouseenter', () => { comboActiveIndex = idx; highlightActiveItem(); });
    panel.appendChild(item);
  });
  highlightActiveItem();
}

function highlightActiveItem() {
  document.querySelectorAll('#modelComboPanel .combo-item').forEach((el) => {
    const isActive = Number(el.dataset.idx) === comboActiveIndex;
    el.classList.toggle('active', isActive);
    if (isActive) el.scrollIntoView({ block: 'nearest' });
  });
}

function openComboPanel() {
  renderComboPanel(document.getElementById('modelSearch').value.trim());
  document.getElementById('modelComboPanel').style.display = 'block';
}

function closeComboPanel() {
  document.getElementById('modelComboPanel').style.display = 'none';
}

const MODEL_TYPE_COLORS = {
  ctq_anima: 'type-anima',
  ctq_flux2: 'type-flux2',
  flux1_custom: 'type-flux1',
  sdxl_custom: 'type-sdxl',
  qwen21_custom: 'type-qwen', ctq_qwen: 'type-qwen',
  ctq_zimage: 'type-zimage', ctq_zimage_refiner: 'type-zimage',
  ctq_wan: 'type-video', ctq_hunyuan: 'type-video', ctq_minimaxh3: 'type-video', ctq_ltxv2: 'type-video',
  ctq_krea2: 'type-other', ctq_boogu: 'type-other', ctq_ideogram4: 'type-other', ctq_radiance: 'type-other',
  ctq_nerf_large: 'type-other', ctq_nerf_small: 'type-other',
  ctq_distillation_large: 'type-other', ctq_distillation_small: 'type-other',
  ctq_gemma4: 'type-other', ctq_qwen_vlm: 'type-other',
};
let modelTypeDetectSeq = 0;

async function detectAndApplyModelType(m) {
  const pill = document.getElementById('modelTypePill');
  const mySeq = ++modelTypeDetectSeq;
  pill.className = 'pill loading';
  pill.innerHTML = '<span class="dot"></span><span>' + t('modelTypeDetecting') + '</span>';
  pill.style.display = 'inline-flex';
  try {
    const res = await fetch('/api/detect_model?path=' + encodeURIComponent(m.path));
    const data = await res.json();
    if (mySeq !== modelTypeDetectSeq) return; // Nutzer hat inzwischen ein anderes Modell gewaehlt
    if (!res.ok || !data.preset_key || !EXCLUDES[data.preset_key]) {
      pill.style.display = 'none';
      return;
    }
    const colorClass = MODEL_TYPE_COLORS[data.preset_key] || 'type-other';
    const label = EXCLUDES[data.preset_key].chip || formatLabelFor(EXCLUDES[data.preset_key]);
    pill.className = 'pill ' + colorClass + ' pop';
    pill.innerHTML = '<span class="dot"></span><span>' + t('modelTypeDetectedTemplate', escapeHtml(label)) + '</span>';
    applyExcludePreset(data.preset_key);
  } catch (e) {
    if (mySeq === modelTypeDetectSeq) pill.style.display = 'none';
  }
}

function selectModel(m) {
  selectedModel = m;
  document.getElementById('modelSearch').value = modelDisplayName(m);
  document.getElementById('inputPath').value = m.path;
  const preview = document.getElementById('modelPathPreview');
  preview.innerHTML = t('fullPathLabel') + '<code>' + escapeHtml(m.path) + '</code> <span class="badge info">'
    + escapeHtml(m.root_label) + '</span>';
  preview.style.display = 'block';
  closeComboPanel();
  updateActionSummary();
  detectAndApplyModelType(m);
}

function updateActionSummary() {
  const el = document.getElementById('actionSummary');
  const path = document.getElementById('inputPath').value.trim();
  if (!path) {
    el.innerHTML = t('chooseModelFirst');
    return;
  }
  const name = path.split(/[\\\/]/).pop();
  const fmtKey = document.getElementById('formatSelect').value;
  const fmt = FORMATS[fmtKey];
  const rawLabel = fmt ? formatLabelFor(fmt) : fmtKey;
  const fmtLabel = rawLabel.replace(/\s*\([^)]*\)/g, '').trim();
  el.innerHTML = '<b>' + escapeHtml(name) + '</b> &rarr; ' + escapeHtml(fmtLabel);
}

function showManualPath(show) {
  document.getElementById('manualPathRow').style.display = show ? 'block' : 'none';
  document.getElementById('manualPathToggle').textContent = show ? t('manualToggleHide') : t('manualToggleShow');
}

async function loadLogsList() {
  const box = document.getElementById('logsList');
  try {
    const res = await fetch('/api/logs');
    const data = await res.json();
    const logs = data.logs || [];
    LOGS_COUNT = logs.length;
    box.innerHTML = '';
    if (!logs.length) {
      box.textContent = t('logsNone');
    } else {
      const locale = LANG === 'en' ? 'en-US' : 'de-DE';
      for (const entry of logs) {
        const row = document.createElement('div');
        row.className = 'log-row';
        const info = document.createElement('span');
        const dt = new Date(entry.mtime * 1000);
        info.textContent = dt.toLocaleString(locale) + '   ' + entry.name + '   (' + Math.max(1, Math.round(entry.size / 1024)) + ' KB)';
        const btn = document.createElement('button');
        btn.type = 'button'; btn.className = 'secondary'; btn.textContent = t('logsView');
        btn.addEventListener('click', () => viewLog(entry.name));
        row.appendChild(info); row.appendChild(btn);
        box.appendChild(row);
      }
    }
  } catch (e) {
    box.textContent = t('logsError');
  }
  updateStatsBar();
}

async function viewLog(name) {
  const viewer = document.getElementById('logViewer');
  viewer.style.display = 'block';
  viewer.textContent = t('logsLoading', name);
  try {
    const res = await fetch('/api/logs/' + encodeURIComponent(name));
    viewer.textContent = res.ok ? await res.text() : t('logsLoadError');
  } catch (e) {
    viewer.textContent = t('logsLoadError');
  }
  viewer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateStatsBar() {
  document.getElementById('statsBar').textContent = t('statsBarTemplate', MODEL_ROOTS_COUNT, MODELS.length, LOGS_COUNT);
}

function updateFormatHint() {
  const key = document.getElementById('formatSelect').value;
  const fmt = FORMATS[key];
  const hint = document.getElementById('formatHint');
  const badge = fmt.verified
    ? '<span class="badge ok">' + t('badgeVerified') + '</span>'
    : '<span class="badge warn">' + t('badgeExperimental') + '</span>';
  const verifiedText = fmt.verified ? t('formatVerifiedText') : t('formatUnverifiedText');
  const desc = LANG === 'en' ? (fmt.description_en || fmt.description) : fmt.description;
  hint.innerHTML = badge + ' ' + desc + ' ' + verifiedText;
  document.getElementById('groupsizeRow').style.display =
    (key === 'int8_convrot') ? 'block' : 'none';
  document.getElementById('blockSizeRow').style.display =
    (key === 'int8_block') ? 'block' : 'none';
  updateActionSummary();
}

function updateExcludeHint() {
  const key = document.getElementById('excludeSelect').value;
  const preset = EXCLUDES[key];
  const hint = document.getElementById('excludeHint');
  if (!preset) { hint.textContent = ''; return; }
  if (preset.kind === 'regex') {
    const badge = preset.verified
      ? '<span class="badge ok">' + t('badgeVerified') + '</span>'
      : '<span class="badge warn">' + t('badgeCommunityUnverified') + '</span>';
    hint.innerHTML = badge + ' ' + t('excludeActivePattern') + '<code>' + preset.value + '</code>'
      + t('excludeUnquantizedNote')
      + (preset.verified ? '' : t('excludeCheckComfyUI'));
  } else if (preset.kind === 'builtin') {
    const badge = preset.verified ? '<span class="badge ok">' + t('badgeEndToEnd') + '</span> ' : '';
    let desc;
    if (preset.filter_desc) {
      const note = LANG === 'en' ? preset.filter_desc_note_en : preset.filter_desc_note_de;
      desc = '<code>' + preset.value + '</code>: ' + preset.filter_desc + (note ? ' ' + note : '') + '.';
    } else {
      desc = t('excludeUsesBuiltin') + '<code>' + preset.value + '</code>.';
    }
    hint.innerHTML = badge + desc + t('excludeFullDetails');
  } else if (preset.kind === 'none') {
    hint.textContent = t('excludeNoneText');
  } else if (preset.kind === 'custom') {
    hint.textContent = t('excludeCustomText');
  } else {
    hint.textContent = '';
  }
  updateChipActive();
  updateAdvancedHint();
}

function updateAdvancedHint() {
  const key = document.getElementById('excludeSelect').value;
  const preset = EXCLUDES[key];
  const box = document.getElementById('advancedPresetHint');
  if (!preset || (preset.kind !== 'builtin' && preset.kind !== 'regex')) {
    box.style.display = 'none';
    return;
  }
  box.style.display = 'block';
  if (preset.kind === 'builtin') {
    box.innerHTML = t('advancedBuiltinText', preset.value);
  } else {
    box.innerHTML = t('advancedRegexText');
  }
}

async function startJob() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.textContent = t('runBtnRunning');
  document.body.classList.add('job-active');
  CURRENT_JOB_ID = null;

  const body = {
    input_path: document.getElementById('inputPath').value,
    output_path: document.getElementById('outputPath').value,
    format: document.getElementById('formatSelect').value,
    exclude_preset: document.getElementById('excludeSelect').value,
    custom_regex: document.getElementById('customRegex').value,
    convrot_groupsize: document.getElementById('convrotGroupsize').value,
    block_size: document.getElementById('blockSize').value,
    low_memory_mode: document.getElementById('lowMemoryMode').value,
    extra_args: document.getElementById('extraArgs').value,
  };

  document.getElementById('log').textContent = '';
  document.getElementById('jobLogLink').style.display = 'none';
  document.getElementById('actionProgress').style.display = 'flex';
  const bar = document.getElementById('progressBar');
  bar.style.width = '0%';
  bar.classList.add('running');
  setStatusLine(t('startingStatus'), '');
  openDrawer();
  startTimer();

  let data;
  try {
    const res = await fetch('/api/quantize', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    data = await res.json();
    if (!res.ok) {
      setStatusLine(t('errorStatusTemplate', data.error), 'status-error');
      bar.classList.remove('running');
      stopTimer();
      resetButton();
      return;
    }
  } catch (e) {
    // z.B. Server abgestuerzt/nicht erreichbar -- ohne das wuerde der Button
    // fuer immer deaktiviert bleiben, ohne jede Rueckmeldung.
    setStatusLine(t('errorStatusTemplate', String((e && e.message) || e)), 'status-error');
    bar.classList.remove('running');
    stopTimer();
    resetButton();
    return;
  }

  CURRENT_JOB_ID = data.job_id;
  poll(data.job_id);
}

async function cancelJob() {
  if (!CURRENT_JOB_ID || !confirm(t('cancelConfirm'))) return;
  const btns = document.querySelectorAll('.cancel-job-btn');
  btns.forEach((b) => { b.disabled = true; b.textContent = t('cancellingBtn'); });
  try {
    const res = await fetch('/api/cancel/' + CURRENT_JOB_ID, { method: 'POST' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      alert(data.error || t('cancelFailed'));
    }
  } catch (e) {
    alert(t('cancelFailed'));
  }
  // Der Status "cancelled" kommt ueber die normale Abfrageschleife herein.
  btns.forEach((b) => { b.disabled = false; b.textContent = t('cancelJobBtn'); });
}

function setStatusLine(text, cls) {
  const el = document.getElementById('drawerStatusLine');
  el.textContent = text;
  el.className = 'job-drawer-status ' + cls;
  const reopenBtn = document.getElementById('drawerReopenBtn');
  document.getElementById('drawerReopenText').textContent = text;
  reopenBtn.className = 'drawer-reopen-pill ' + cls;
  document.getElementById('jobDrawer').classList.toggle('running', cls === '' || cls === 'status-running');
}

function openDrawer() {
  document.getElementById('jobDrawer').classList.add('open');
  document.getElementById('drawerReopenBtn').style.display = 'none';
}

function minimizeDrawer() {
  document.getElementById('jobDrawer').classList.remove('open');
  document.getElementById('drawerReopenBtn').style.display = 'inline-flex';
}

let jobStartTime = null;
let timerInterval = null;

function startTimer() {
  jobStartTime = Date.now();
  clearInterval(timerInterval);
  timerInterval = setInterval(updateTimerDisplay, 500);
  updateTimerDisplay();
}

function stopTimer() {
  clearInterval(timerInterval);
}

function updateTimerDisplay() {
  if (!jobStartTime) return;
  const secs = Math.floor((Date.now() - jobStartTime) / 1000);
  const m = Math.floor(secs / 60), s = secs % 60;
  document.getElementById('drawerTimer').textContent = t('durationLabel') + m + ':' + String(s).padStart(2, '0');
}

function poll(jobId) {
  // Eigene Generation statt setInterval: verhindert, dass sich ueberlappende
  // Requests (falls eine Antwort mal laenger als 1200ms braucht) gegenseitig
  // ueberholen und den Log/Fortschritt mit einer veralteten Antwort ueberschreiben --
  // der naechste Poll wird immer erst nach Abschluss des vorherigen geplant.
  const myGeneration = ++pollGeneration;

  async function tick() {
    if (myGeneration !== pollGeneration) return;

    let data;
    try {
      const res = await fetch('/api/status/' + jobId);
      data = await res.json();
    } catch (e) {
      if (myGeneration === pollGeneration) setTimeout(tick, 1200);
      return;
    }
    if (myGeneration !== pollGeneration) return;

    const bar = document.getElementById('progressBar');
    bar.style.width = (data.progress || 0) + '%';
    document.getElementById('log').textContent = (data.log || []).join('\n');
    const logEl = document.getElementById('log');
    logEl.scrollTop = logEl.scrollHeight;

    const jobLogLink = document.getElementById('jobLogLink');
    if (data.log_file) {
      jobLogLink.style.display = 'block';
      jobLogLink.innerHTML = t('fullLogSavingAs') + '<a href="#" id="jobLogLinkA">' + data.log_file + '</a>';
      const a = document.getElementById('jobLogLinkA');
      if (a) a.addEventListener('click', (e) => { e.preventDefault(); viewLog(data.log_file); });
    }

    if (data.status === 'running' || data.status === 'starting') {
      bar.classList.add('running');
      setStatusLine(t('runningStatusTemplate', data.progress || 0), 'status-running');
      setTimeout(tick, 1200);
    } else if (data.status === 'done') {
      bar.classList.remove('running');
      setStatusLine(t('doneStatusTemplate', data.output_path), 'status-done');
      stopTimer();
      resetButton();
      loadLogsList();
    } else if (data.status === 'error') {
      bar.classList.remove('running');
      setStatusLine(t('errorStatusTemplate', data.error), 'status-error');
      stopTimer();
      resetButton();
      loadLogsList();
    } else if (data.status === 'cancelled') {
      bar.classList.remove('running');
      setStatusLine(t('cancelledStatus'), 'status-cancelled');
      stopTimer();
      resetButton();
      loadLogsList();
    } else {
      setTimeout(tick, 1200);
    }
  }

  tick();
}

function resetButton() {
  const btn = document.getElementById('runBtn');
  btn.disabled = false; btn.textContent = t('runBtn');
  document.body.classList.remove('job-active');
}

async function showCtqHelp() {
  const btn = document.getElementById('ctqHelpBtn');
  const out = document.getElementById('ctqHelpOutput');
  if (out.style.display !== 'none') {
    out.style.display = 'none';
    btn.textContent = t('ctqHelpShow');
    return;
  }
  btn.textContent = t('ctqHelpLoading');
  const res = await fetch('/api/ctq_help');
  const data = await res.json();
  out.textContent = data.help || t('errorStatusTemplate', data.error);
  out.style.display = 'block';
  btn.textContent = t('ctqHelpHide');
}

document.getElementById('runBtn').addEventListener('click', startJob);
document.getElementById('cancelJobBarBtn').addEventListener('click', cancelJob);
document.getElementById('cancelJobDrawerBtn').addEventListener('click', cancelJob);
document.getElementById('ctqHelpBtn').addEventListener('click', showCtqHelp);
document.getElementById('customRegexRow').style.display = 'none';
document.getElementById('refreshModelsBtn').addEventListener('click', loadModels);
document.getElementById('manualPathToggle').addEventListener('click', () => {
  const showing = document.getElementById('manualPathRow').style.display !== 'none';
  showManualPath(!showing);
});
document.getElementById('inputPath').addEventListener('input', updateActionSummary);

const modelSearchEl = document.getElementById('modelSearch');
modelSearchEl.addEventListener('focus', openComboPanel);
modelSearchEl.addEventListener('input', openComboPanel);
modelSearchEl.addEventListener('keydown', (e) => {
  const panelOpen = document.getElementById('modelComboPanel').style.display !== 'none';
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (!panelOpen) { openComboPanel(); return; }
    if (comboActiveIndex < comboVisibleItems.length - 1) comboActiveIndex++;
    highlightActiveItem();
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (comboActiveIndex > 0) comboActiveIndex--;
    highlightActiveItem();
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (comboActiveIndex >= 0 && comboVisibleItems[comboActiveIndex]) {
      selectModel(comboVisibleItems[comboActiveIndex]);
    }
  } else if (e.key === 'Escape') {
    closeComboPanel();
    modelSearchEl.blur();
  }
});
modelSearchEl.addEventListener('blur', () => {
  setTimeout(() => {
    closeComboPanel();
    if (selectedModel) {
      modelSearchEl.value = modelDisplayName(selectedModel);
    }
  }, 150);
});
document.getElementById('drawerCloseBtn').addEventListener('click', minimizeDrawer);
document.getElementById('drawerReopenBtn').addEventListener('click', openDrawer);

(function setupJobDrawerDragAndResize() {
  const drawer = document.getElementById('jobDrawer');
  const header = document.getElementById('jobDrawerHeader');
  const handle = document.getElementById('jobDrawerResizeHandle');

  function clamp(v, min, max) { return Math.min(Math.max(v, min), max); }

  // Von der Zentrierung (translate -50%) auf feste left/top-Pixel umstellen,
  // damit Ziehen und Vergroessern an der oberen linken Ecke verankert sind.
  function pinToCurrentPosition() {
    const rect = drawer.getBoundingClientRect();
    drawer.style.left = rect.left + 'px';
    drawer.style.top = rect.top + 'px';
    drawer.classList.add('placed');
    return rect;
  }

  let dragging = false, dragStartX = 0, dragStartY = 0, dragStartLeft = 0, dragStartTop = 0;

  header.addEventListener('mousedown', (e) => {
    if (e.target.closest('button')) return;
    dragging = true;
    drawer.classList.add('no-anim');
    const rect = pinToCurrentPosition();
    document.body.classList.add('drawer-noselect');
    dragStartX = e.clientX; dragStartY = e.clientY;
    dragStartLeft = rect.left; dragStartTop = rect.top;
    e.preventDefault();
  });

  let resizing = false, resizeStartX = 0, resizeStartY = 0, resizeStartW = 0, resizeStartH = 0;

  handle.addEventListener('mousedown', (e) => {
    resizing = true;
    drawer.classList.add('no-anim');
    const rect = pinToCurrentPosition();
    resizeStartX = e.clientX; resizeStartY = e.clientY;
    resizeStartW = rect.width; resizeStartH = rect.height;
    document.body.classList.add('drawer-noselect');
    e.preventDefault();
    e.stopPropagation();
  });

  window.addEventListener('mousemove', (e) => {
    if (dragging) {
      const dx = e.clientX - dragStartX, dy = e.clientY - dragStartY;
      const maxLeft = Math.max(window.innerWidth - drawer.offsetWidth, 0);
      const maxTop = Math.max(window.innerHeight - drawer.offsetHeight, 0);
      drawer.style.left = clamp(dragStartLeft + dx, 0, maxLeft) + 'px';
      drawer.style.top = clamp(dragStartTop + dy, 0, maxTop) + 'px';
    } else if (resizing) {
      const dx = e.clientX - resizeStartX, dy = e.clientY - resizeStartY;
      const rect = drawer.getBoundingClientRect();
      const maxW = window.innerWidth - rect.left - 12;
      const maxH = window.innerHeight - rect.top - 12;
      drawer.style.maxWidth = 'none';
      drawer.style.maxHeight = 'none';
      drawer.style.width = clamp(resizeStartW + dx, 300, maxW) + 'px';
      drawer.style.height = clamp(resizeStartH + dy, 180, maxH) + 'px';
    }
  });

  window.addEventListener('mouseup', () => {
    if (dragging || resizing) {
      dragging = false; resizing = false;
      drawer.classList.remove('no-anim');
      document.body.classList.remove('drawer-noselect');
    }
  });
})();

applyTheme();
document.getElementById('themeToggleBtn').addEventListener('click', toggleTheme);
applyMinimal();
document.getElementById('minimalToggleBtn').addEventListener('click', toggleMinimal);
document.getElementById('langSelect').value = LANG;
document.getElementById('langSelect').addEventListener('change', (e) => setLanguage(e.target.value));
applyStaticI18n();

document.getElementById('settingsBtn').addEventListener('click', () => {
  if (settingsIsOpen()) closeSettings(); else openSettings(false);
});
document.getElementById('ctqStatus').addEventListener('click', () => { if (!CTQ_FOUND) openSettings(false); });
document.getElementById('settingsCloseBtn').addEventListener('click', closeSettings);
document.getElementById('settingsCancelBtn').addEventListener('click', closeSettings);
document.getElementById('settingsSaveBtn').addEventListener('click', saveSettings);
document.addEventListener('mousedown', (e) => {
  if (settingsIsOpen() && !e.target.closest('.settings-anchor')) closeSettings();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && settingsIsOpen()) closeSettings();
});
document.getElementById('addRootBtn').addEventListener('click', async () => {
  const path = await pickNativeFolder(t('settingsAddRoot'), '');
  if (path) addModelRootFromPath(path);
});
document.getElementById('browseComfyuiBtn').addEventListener('click', async () => {
  const path = await pickNativeFolder(t('settingsComfyuiHeading'), document.getElementById('comfyuiRootInput').value);
  if (path) document.getElementById('comfyuiRootInput').value = path;
});
document.getElementById('browseCtqBtn').addEventListener('click', async () => {
  const path = await pickNativeFile(
    t('settingsCtqHeading'), 'ctq executable|ctq.exe|All files|*.*',
    document.getElementById('ctqOverrideInput').value
  );
  if (path) document.getElementById('ctqOverrideInput').value = path;
});
document.getElementById('runSystemCheckBtn').addEventListener('click', runSystemCheck);

loadConfig();
loadModels();
loadLogsList();

fetch('/api/settings').then((r) => r.json()).then((data) => {
  if (data.first_run && !readStored('setupSeen', '')) {
    writeStored('setupSeen', '1');
    openSettings(true);
  }
}).catch(() => {});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("ctq gefunden unter:", CTQ_EXE or "NICHT GEFUNDEN")
    print("Oeffne im Browser: http://127.0.0.1:8877")
    # threaded=True, damit ein offener nativer Ordner-/Datei-Dialog (blockiert
    # den Request, bis der Nutzer waehlt/abbricht) nicht den ganzen Server
    # einfriert -- ein parallel laufender Quantisierungs-Job soll trotzdem
    # weiter Status liefern koennen.
    app.run(host="127.0.0.1", port=8877, debug=False, threaded=True)
