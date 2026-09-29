<p align="center">
  <img src="icon.png" alt="GFlava-Quant logo" width="120">
</p>

<h1 align="center">GFlava-Quant</h1>

<p align="center">
  A local, single-file web UI for quantizing ComfyUI diffusion models with
  <a href="https://github.com/silveroxides/convert_to_quant"><code>convert_to_quant</code></a> (<code>ctq</code>).
</p>

<p align="center">
  <em><a href="README.de.md">Diese Seite auf Deutsch lesen</a></em>
</p>

<p align="center">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-lightgrey">
  <img alt="Python 3" src="https://img.shields.io/badge/python-3.10%2B-blue">
</p>

---

Quantizing diffusion checkpoints for ComfyUI usually means memorizing a pile
of `ctq` CLI flags, remembering which layers each architecture needs kept in
full precision, and manually patching the `.comfy_quant` metadata afterwards
so the file loads without warnings. GFlava-Quant wraps all of that in a
small local web app: pick a model, let it recognize the architecture, click
quantize, watch the log.

It's a single Python file (Flask + vanilla JS, no build step, no framework)
that runs only on `127.0.0.1` — nothing leaves your machine.

## Features

- **Automatic model architecture detection.** Selecting a model reads just
  its safetensors header (no tensor data, so it's fast even on huge files)
  and matches the tensor names against known architecture signatures. On a
  confident match, a colored pill shows the detected type and the correct
  layer-exclusion preset is applied automatically. Detects Anima, Flux.1,
  Flux.2, SDXL/Illustrious, Qwen-Image 2.1 (single-stream), ctq's older
  Qwen (dual-stream), Z-Image (+ Refiner), Wan, HunyuanVideo, Krea2, Boogu,
  Ideogram4, Radiance, NeRF (large/small), Chroma/distilled (large/small),
  MinimaxH3, LTXv2, Gemma4 and Qwen3-VL. Several signatures are verified
  against real model files, not just documentation. No confident match?
  No pill — you pick manually, same as before.
- **Every layer-exclusion preset ctq knows about**, pulled straight from
  your installed `ctq --help-filters`, plus three hand-verified community
  regexes for architectures ctq doesn't have a built-in flag for
  (SDXL/Illustrious, Flux.1, Qwen-Image 2.1 single-stream). Available both
  as quick-select chips and in the dropdown — 27 presets in total.
- **Six quantization formats**: INT8 ConvRot, INT8 Tensor-wise, INT8
  Block-wise, FP8, NVFP4 and MXFP8, with format-specific fields (ConvRot
  group size, block size) and inline documentation sourced from ctq's own
  `--help` output.
- **Automatic `.comfy_quant` marker fix** after INT8 ConvRot/Tensor-wise
  runs, so the output loads in ComfyUI without the `unet unexpected`
  warning.
- **Smart low-memory mode.** Controls ctq's `--low-memory` flag, which
  governs system RAM usage while loading (verified against ctq's installed
  source, not just its `--help` text) — it has nothing to do with VRAM, ctq
  already streams one tensor at a time onto the GPU regardless. "Automatic"
  only enables it when the input file is larger than 50% of currently
  available RAM (ctq's own recommendation); the measured values and
  decision are logged for every run.
- **Persistent, per-job logs.** Every run's full command and output is
  saved to `logs/` and stays there across server restarts, browsable from
  the UI.
- **Live job window** that opens centered (1000×500, draggable and
  resizable) with status and streaming log; progress bar and elapsed time
  sit in the bottom bar. Minimizes to a small status pill so you can keep
  working.
- **Cancel a running job.** Stops `ctq` together with its Python child
  process (otherwise the GPU would keep working) and removes the incomplete
  output file — but only if that file didn't exist before the job started.
- **Configurable model folders, no hardcoding.** Add or remove scanned
  folders from the Settings dropdown using real native Windows folder/file
  dialogs (not a custom in-app browser) — saved to `config.json`, reloaded
  automatically on the next start. On first launch the list is empty and
  Settings opens with a short hint. Includes real-time search across all
  configured folders.
- **System check.** Verifies whether `ctq` is installed and up to date
  (compares against PyPI), and whether the optional
  [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast)
  custom node is installed and current (compares local vs. remote git
  revision) — with one-click install/update buttons.
- **Dark/light theme, minimal design mode and German/English UI**, all
  persisted in the browser and switched without a page reload. The minimal
  mode drops every gradient and uses a single accent color.
- **Fully local, including fonts.** Space Grotesk, IBM Plex Sans and
  JetBrains Mono are bundled under `fonts/` — the page makes no requests to
  the internet.
- **One-click launcher** (`start_server.bat`): finds `python.exe` and
  `quant_server.py` automatically (or lets you pick them via a native file
  dialog), remembers the choice per machine, and offers to install `flask`
  and `convert_to_quant` if either is missing — with a plain-language
  explanation of what each one is for before it asks.

## Requirements

- Windows, with a Python environment that already runs ComfyUI (this was
  built against a ComfyUI portable / `python_embeded` install, but any
  Python 3.10+ environment works).
- [`convert_to_quant`](https://github.com/silveroxides/convert_to_quant)
  (`ctq`) and [`flask`](https://flask.palletsprojects.com/) — both can be
  installed for you by `start_server.bat`, or manually:

  ```powershell
  python.exe -m pip install flask convert_to_quant
  ```

## Quickstart

1. Copy `quant_server.py`, `start_server.bat` and `icon.png` into (or next
   to) your ComfyUI Python environment.
2. Double-click `start_server.bat`. First run: it locates or asks you to
   pick `python.exe`, checks/installs `flask` and `convert_to_quant`, then
   starts the server and opens your browser at `http://127.0.0.1:8877`.
3. Open the gear icon → Settings and add your model folders (native folder
   picker). That's the whole setup.

Prefer to run it by hand?

```powershell
python.exe quant_server.py
```

## Usage

1. **Select a model** from the searchable dropdown (populated from your
   configured folders) or type a path manually. If the architecture is
   recognized, a colored pill appears and the right layer-exclusion preset
   is pre-selected.
2. **Pick a quantization format.** INT8 ConvRot is the default and the only
   format we've personally verified end-to-end in ComfyUI; the others are
   implemented per ctq's own documentation but not independently verified
   by us — check the result in ComfyUI before relying on it.
3. **Check the layer-exclusion preset** — auto-selected if detected,
   otherwise pick from the chips or dropdown. Hover/read the hint text for
   ctq's own description of what each preset keeps at full precision.
4. **(Optional) Advanced**: extra raw ctq arguments, and the low-memory
   mode override (Automatic / Always on / Always off).
5. **Start quantization.** A centered job window shows live status and log
   output until the job finishes or errors; it keeps running if you
   minimize it. Use **Cancel** (in the job window or the bottom bar) to
   stop a running job.

## Configuration

Everything's stored next to `quant_server.py`, created automatically on
first run, and safe to back up or edit by hand:

- `config.json` — model folders, ComfyUI installation path, optional `ctq`
  path override.
- `start_server_config.txt` — the `python.exe` / `quant_server.py` paths
  the launcher remembered for this machine. Delete it to re-pick.
- `logs/` — one file per quantization run.

None of these are meant to be shared between machines — each is
regenerated automatically, which is also why they're gitignored here.

## Limitations

- Only one quantization job runs at a time (by design — GPU/RAM
  contention).
- Format-level correctness beyond INT8 ConvRot hasn't been independently
  verified against official reference files — always sanity-check output
  in ComfyUI.
- Windows-only (uses `System.Windows.Forms` dialogs via PowerShell for
  native folder/file pickers, and Windows APIs for RAM detection).

## Acknowledgments

GFlava-Quant is a thin UI layer — the real quantization work is done by:

- **[convert_to_quant (ctq)](https://github.com/silveroxides/convert_to_quant)**
  by [silveroxides](https://github.com/silveroxides) — the quantization
  engine this entire tool wraps and calls out to.
- **[ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast)**
  by [BobJohnson24](https://github.com/BobJohnson24) — optional companion
  custom node for fast INT8 inference, integrated into the Settings system
  check.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by
  [comfyanonymous](https://github.com/comfyanonymous) — the platform this
  tool produces models for.
- **[Flask](https://flask.palletsprojects.com/)** (Pallets Projects) —
  the local web server.
- **[safetensors](https://github.com/huggingface/safetensors)** (Hugging
  Face) and **[PyTorch](https://pytorch.org/)** — tensor format and
  runtime that `ctq` and the marker-fix step build on.
- **[Space Grotesk](https://github.com/floriankarsten/space-grotesk)**
  (Florian Karsten and The Space Grotesk Project Authors),
  **[IBM Plex Sans](https://github.com/IBM/plex)** (IBM) and
  **[JetBrains Mono](https://github.com/JetBrains/JetBrainsMono)**
  (JetBrains) — the UI typefaces, bundled under the SIL Open Font License
  1.1 (see `fonts/OFL.txt`).

This tool was designed and built collaboratively with
[Claude Code](https://claude.com/claude-code).

## License

MIT — see [LICENSE](LICENSE). This project is not affiliated with
`convert_to_quant`, ComfyUI, or any of the projects it wraps or
integrates with.
