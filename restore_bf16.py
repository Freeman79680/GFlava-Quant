"""Turns quantized models (INT8 ConvRot from ctq, INT6 "W6A8") back into a plain bf16/fp16/fp32 .safetensors file.

Meant for workflows where a quantized model has to be a normal full-precision model again,
for example to merge it with other models. What it does, per quantized layer:

  INT8 (format int8_tensorwise)                    INT6 (format w6a8_int8, packed 6-bit codes)
    weight (int8) * weight_scale                     unpack the 6-bit codes, (code - 32) * s_rel per group of 16,
                                                     round to the INT8 grid, * s_channel per row
    @ Hadamard matrix of the group (ConvRot)         @ Hadamard matrix of the group (ConvRot)
    cast to the chosen dtype                         cast to the chosen dtype

The Hadamard matrix is symmetric and orthogonal, so the rotation is its own inverse. These are the
formulas ComfyUI itself uses when it loads such files (comfy_kitchen), but computed in float32 with
TF32 switched off and rounded to the output dtype only once at the very end. The rounding of the
quantization cannot be undone: expect about 0.9 % relative deviation from the original weights for
INT8 and clearly more for INT6, no matter how the conversion is done.

Everything that was not quantized is copied byte for byte. The helper tensors of quantized
layers (scales, markers), the per-layer attention config ("comfy_attention.config") and the
quantization metadata are dropped. The output is written tensor by tensor, so the model never has
to fit into RAM as a whole, and under a temporary name until it is complete.

Header reading and scanning need no torch (the server uses them); only the conversion does.

    python restore_bf16.py -i model_int8.safetensors -o model_bf16.safetensors [--dtype bf16]
"""
import argparse
import json
import math
import os
import struct
import sys
import time

SUPPORTED_FORMATS = ("int8_tensorwise", "w6a8_int8")   # ctq's INT8 (with or without ConvRot); 6-bit weights, 8-bit activations
DROPPED_SUFFIXES = (".comfy_attention.config",)       # ComfyUI runtime hints of a quantized model, not weights
QUANT_META_KEY = "_quantization_metadata"
OUTPUT_DTYPES = {"bf16": ("BF16", 2, "bfloat16"), "fp16": ("F16", 2, "float16"), "fp32": ("F32", 4, "float32")}
ST_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
            "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
MAX_HEADER = 200_000_000
CHUNK = 64 * 1024 * 1024


class RestoreError(Exception):
    """A problem the user can understand (bad file, unsupported format, ...)."""


# ---------------------------------------------------------------------------
# Reading (no torch)
# ---------------------------------------------------------------------------


def read_header(path):
    """Returns (header, data_start) of a .safetensors file."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise RestoreError("Keine gueltige .safetensors-Datei (zu kurz).")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > MAX_HEADER:
            raise RestoreError("Keine gueltige .safetensors-Datei (Header ungueltig).")
        try:
            header = json.loads(f.read(size))
        except ValueError:
            raise RestoreError("Keine gueltige .safetensors-Datei (Header nicht lesbar).")
    return header, 8 + size


def _nbytes(entry):
    return int(entry["data_offsets"][1]) - int(entry["data_offsets"][0])


def _numel(shape):
    return math.prod(int(d) for d in shape)


def _read_range(f, data_start, entry):
    f.seek(data_start + int(entry["data_offsets"][0]))
    return f.read(_nbytes(entry))


def _is_power_of_4(n):
    return isinstance(n, int) and n >= 4 and n & (n - 1) == 0 and (n.bit_length() - 1) % 2 == 0


def _name_index(keys, suffix):
    """The base names of all tensors called <base><suffix>, and an index of every dotted tail of them."""
    bases = {k[: -len(suffix)] for k in keys if k.endswith(suffix)}
    by_suffix = {}
    for base in bases:
        parts = base.split(".")
        for i in range(len(parts)):
            by_suffix.setdefault(".".join(parts[i:]), []).append(base)
    return bases, by_suffix


def _find_weight_base(marker_base, bases, by_suffix):
    """Which tensor name belongs to a layer description? Normally the same one; some converters write the
    descriptions without the 'model.diffusion_model.' prefix the weights carry (or the other way round)."""
    if marker_base in bases:
        return marker_base
    hits = by_suffix.get(marker_base, [])          # the weights carry an extra prefix
    if len(hits) == 1:
        return hits[0]
    parts = marker_base.split(".")
    for i in range(1, len(parts)):                 # the description carries an extra prefix
        if ".".join(parts[i:]) in bases:
            return ".".join(parts[i:])
    return None


def _layer_descriptions(f, header, data_start, keys, metadata, problems):
    """[(layer name, description dict, marker tensor key or None)]. The description is read from the small
    <layer>.comfy_quant tensor; layers without one (some formats only list them in the file's
    _quantization_metadata) come from that metadata."""
    found, names = [], set()
    for key in sorted(k for k in keys if k.endswith(".comfy_quant")):
        name = key[: -len(".comfy_quant")]
        try:
            description = json.loads(_read_range(f, data_start, header[key]).rstrip(b"\0").decode("utf-8"))
        except Exception:
            problems.append(f"{name}: Quantisierungs-Marker nicht lesbar")
            continue
        found.append((name, description, key))
        names.add(name)
    try:
        listed = json.loads(metadata.get(QUANT_META_KEY) or "{}").get("layers") or {}
    except (ValueError, AttributeError):
        listed = {}
    for name, description in listed.items():
        if name not in names and isinstance(description, dict):
            found.append((name, description, None))
    return found


def scan(path):
    """Looks at a model file without loading tensors. Returns a dict:

    layers       {weight tensor base name: layer info} of all quantized layers that can be converted
    unsupported  {format name: count} of quantized layers in other formats
    problems     list of texts, each makes the file unusable for the conversion
    header, data_start, metadata  for the conversion itself
    skipped      tensor names that are not copied (scales, markers, attention configs)
    tensors, input_bytes, output_bytes {bf16, fp16, fp32}
    """
    header, data_start = read_header(path)
    metadata = header.get("__metadata__") or {}
    keys = [k for k in header if k != "__metadata__"]
    int8_bases, int8_index = _name_index(keys, ".weight_scale")
    w6_bases, w6_index = _name_index(keys, ".weight_s_channel")
    layers, unsupported, problems = {}, {}, []
    claimed_int8, claimed_w6 = set(), set()

    with open(path, "rb") as f:
        descriptions = _layer_descriptions(f, header, data_start, keys, metadata, problems)
    for name, desc, marker_key in descriptions:
        fmt = str(desc.get("format"))
        if fmt not in SUPPORTED_FORMATS:
            unsupported[fmt] = unsupported.get(fmt, 0) + 1
            continue
        if fmt == "int8_tensorwise":
            base = _find_weight_base(name, int8_bases, int8_index)
            scale_key = base + ".weight_scale" if base is not None else None
            rel_key = None
        else:
            base = _find_weight_base(name, w6_bases, w6_index)
            scale_key = base + ".weight_s_channel" if base is not None else None
            rel_key = base + ".weight_s_rel" if base is not None else None
        weight = header.get(base + ".weight") if base is not None else None
        scale = header.get(scale_key) if scale_key else None
        rel = header.get(rel_key) if rel_key else None
        if weight is None or scale is None or (fmt == "w6a8_int8" and rel is None):
            problems.append(f"{name}: Gewicht oder Skala fehlt")
            continue
        if base in layers:                      # the same layer listed twice (marker tensor and metadata)
            continue
        (claimed_int8 if fmt == "int8_tensorwise" else claimed_w6).add(base)
        if weight.get("dtype") != "I8" or len(weight["shape"]) != 2:
            problems.append(f"{name}: Gewicht ist nicht INT8 mit 2 Dimensionen")
            continue
        rows, packed_cols = (int(d) for d in weight["shape"])

        if fmt == "int8_tensorwise":
            cols = packed_cols
            if scale.get("dtype") != "F32" or _numel(scale["shape"]) not in (1, rows):
                problems.append(f"{name}: Skala hat eine unerwartete Form {scale.get('shape')} (erwartet: eine pro Zeile oder eine einzelne)")
                continue
            convrot = bool(desc.get("convrot"))
            group = desc.get("convrot_groupsize") if convrot else None
            if convrot and (not _is_power_of_4(group) or cols % group):
                problems.append(f"{name}: ConvRot-Gruppengroesse {group} passt nicht zu {cols} Eingaengen")
                continue
            layers[base] = {"kind": "int8", "rows": rows, "cols": cols, "convrot": convrot, "group": group,
                            "scale_shape": [int(d) for d in scale["shape"]],
                            "marker_key": marker_key, "scale_key": scale_key, "rel_key": None}
            continue

        # w6a8_int8: packed 6-bit codes [N, 3K/4], one fp8 relative scale per group of group_size values
        # [N, K/group_size], one fp32 scale per row [N]; always rotated with ConvRot.
        group_size = desc.get("group_size", 16)
        group = desc.get("convrot_groupsize", 256)
        if not isinstance(group_size, int) or group_size < 16 or group_size % 16:
            problems.append(f"{name}: Gruppengroesse {group_size} wird nicht unterstuetzt")
            continue
        if len(rel["shape"]) != 2 or int(rel["shape"][0]) != rows or rel.get("dtype") != "F8_E4M3":
            problems.append(f"{name}: relative Skalen haben eine unerwartete Form {rel.get('shape')}")
            continue
        cols = int(rel["shape"][1]) * group_size
        if scale.get("dtype") != "F32" or [int(d) for d in scale["shape"]] != [rows]:
            problems.append(f"{name}: Kanal-Skala hat eine unerwartete Form {scale.get('shape')}")
            continue
        if packed_cols * 8 == cols * 4:
            unsupported["4-Bit-Gewichte (W4A8)"] = unsupported.get("4-Bit-Gewichte (W4A8)", 0) + 1
            continue
        if packed_cols * 8 != cols * 6 or cols % 32:
            problems.append(f"{name}: gepackte Breite {packed_cols} passt nicht zu {cols} Werten bei 6 Bit")
            continue
        if not _is_power_of_4(group) or cols % group:
            problems.append(f"{name}: ConvRot-Gruppengroesse {group} passt nicht zu {cols} Eingaengen")
            continue
        layers[base] = {"kind": "w6a8", "rows": rows, "cols": cols, "convrot": True, "group": group,
                        "group_size": group_size, "scale_shape": [rows],
                        "marker_key": marker_key, "scale_key": scale_key, "rel_key": rel_key}

    # Scales that no description claims belong to an older format that this conversion does not know.
    stray = [b for b in int8_bases if b not in claimed_int8] + [b for b in w6_bases if b not in claimed_w6]
    if stray:
        unsupported["ohne Marker (aelteres Format)"] = len(stray)

    skipped = {k for k in keys if k.endswith(DROPPED_SUFFIXES)}
    for layer in layers.values():
        skipped.update(k for k in (layer["marker_key"], layer["scale_key"], layer["rel_key"]) if k)
    input_bytes = sum(_nbytes(header[k]) for k in keys)
    output_bytes = {}
    for name, (_, size, _) in OUTPUT_DTYPES.items():
        total = 0
        for k in keys:
            if k in skipped:
                continue
            base = k[: -len(".weight")] if k.endswith(".weight") else None
            if base in layers:
                total += layers[base]["rows"] * layers[base]["cols"] * size
            else:
                total += _nbytes(header[k])
        output_bytes[name] = total
    return {"layers": layers, "unsupported": unsupported, "problems": problems, "header": header,
            "data_start": data_start, "metadata": metadata, "tensors": len(keys),
            "input_bytes": input_bytes, "output_bytes": output_bytes, "skipped": skipped}


def summarize(info):
    """The part of scan() that is useful to show (JSON-friendly)."""
    counts = {}
    for layer in info["layers"].values():
        kind = (layer["kind"], layer["convrot"], layer["group"])
        counts[kind] = counts.get(kind, 0) + 1
    kinds = [{"format": fmt, "convrot": convrot, "group": group, "count": n}
             for (fmt, convrot, group), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0]))]
    supported = bool(info["layers"]) and not info["unsupported"] and not info["problems"]
    return {"quantized_layers": len(info["layers"]), "kinds": kinds, "unsupported": info["unsupported"],
            "problems": info["problems"][:5], "tensors": info["tensors"], "input_bytes": info["input_bytes"],
            "output_bytes": info["output_bytes"], "supported": supported}


# ---------------------------------------------------------------------------
# Conversion (needs torch)
# ---------------------------------------------------------------------------

_HADAMARD = {}


def hadamard(size, device):
    """Normalized 'regular' orthogonal Hadamard matrix of ConvRot (Kronecker powers of H4).
    Symmetric and orthogonal, so it is its own inverse."""
    import torch
    key = (size, str(device))
    if key not in _HADAMARD:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h, current = h4, 4
        while current < size:
            h = torch.kron(h, h4)
            current *= 4
        _HADAMARD[key] = (h / math.sqrt(size)).to(device)
    return _HADAMARD[key]


def _rotate(w, group, device):
    rows, cols = w.shape
    h = hadamard(group, device)
    return (w.reshape(rows, cols // group, group) @ h.T).reshape(rows, cols)


def dequantize(q, scale, layer, device):
    """int8 weight + scale -> float32 weight in the original basis (the formula ComfyUI uses)."""
    import torch
    q, scale = q.to(device), scale.to(device)
    w = q.to(torch.float32) * (scale.reshape(-1, 1) if scale.numel() > 1 else scale.reshape(()))
    if layer["convrot"]:
        w = _rotate(w, layer["group"], device)
    return w


def dequantize_w6a8(q, s_rel, s_channel, layer, device):
    """Packed 6-bit weight -> (float32 weight in the original basis, the decoded INT8 levels).

    Storage: [N, 3K/4] bytes = a K/2-byte plane of nibbles (even column = low nibble) followed by a
    K/4-byte plane with each code's top 2 bits (column c: byte c // 4, bit 2 * (c % 4)). The codes are
    uniform, -31..31 stored as +32. Decoding as comfy_kitchen does: (code - 32) * s_rel per group,
    rounded to the INT8 grid; then * s_channel per row and the inverse ConvRot rotation."""
    import torch
    n, k, size = layer["rows"], layer["cols"], layer["group_size"]
    packed = q.to(device).view(torch.uint8).to(torch.int32)
    codes = torch.empty(n, k, dtype=torch.int32, device=device)
    low = packed[:, : k // 2]
    codes[:, 0::2] = low & 0xF
    codes[:, 1::2] = (low >> 4) & 0xF
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
    high = (packed[:, k // 2:].unsqueeze(-1) >> shifts) & 0x3
    codes |= high.reshape(n, k) << 4
    values = (codes.to(torch.float32) - 32.0).view(n, k // size, size) * s_rel.to(device).to(torch.float32).unsqueeze(-1)
    levels = values.view(n, k).round().clamp(-127, 127)
    w = levels * s_channel.to(device).to(torch.float32).view(n, 1)
    return _rotate(w, layer["group"], device), levels


def mismatching_levels(w, levels, s_channel, layer, device):
    """Self-check for the 6-bit format: rotate the restored weight back into the stored basis and divide
    by the row scale; that must give the decoded INT8 levels again."""
    import torch
    s = s_channel.to(device).to(torch.float32).view(-1, 1)
    used = s != 0
    again = torch.round(_rotate(w, layer["group"], device) / torch.where(used, s, torch.ones_like(s)))
    return int(((again != levels) & used).sum().item())


def convert_layer(layer, q, scale, rel, device, verify):
    """-> (float32 weight, number of values the self-check could not reproduce)."""
    if layer["kind"] == "w6a8":
        w, levels = dequantize_w6a8(q, rel, scale, layer, device)
        return w, (mismatching_levels(w, levels, scale, layer, device) if verify else 0)
    w = dequantize(q, scale, layer, device)
    return w, (mismatching_values(w, q, scale, layer, device) if verify else 0)


def mismatching_values(w, q, scale, layer, device):
    """Self-check: quantize the restored weight again with the stored scale and count the INT8
    values that differ from the file. 0 means the restoration lost nothing that was in the file."""
    import torch
    rot = _rotate(w, layer["group"], device) if layer["convrot"] else w
    s = scale.to(device)
    s = s.reshape(-1, 1) if s.numel() > 1 else s.reshape(())
    again = torch.round(rot / s).clamp(-128, 127).to(torch.int8)
    return int((again != q.to(device)).sum().item())


def _to_bytes(t):
    import torch
    try:
        return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    except RuntimeError as e:  # torch without numpy support
        raise RestoreError(f"numpy wird zum Schreiben gebraucht ({e}).")


def _copy(src, dst, count):
    while count > 0:
        chunk = src.read(min(CHUNK, count))
        if not chunk:
            raise RestoreError("Eingabedatei ist unvollstaendig (Ende zu frueh erreicht).")
        dst.write(chunk)
        count -= len(chunk)


def restore(input_path, output_path, dtype="bf16", device="auto", verify=True, log=print):
    """Writes output_path. Returns a dict with counts. Raises RestoreError for problems."""
    import torch
    if dtype not in OUTPUT_DTYPES:
        raise RestoreError(f"Unbekannter Datentyp: {dtype}")
    st_name, size, torch_name = OUTPUT_DTYPES[dtype]
    out_dtype = getattr(torch, torch_name)

    info = scan(input_path)
    if info["problems"]:
        raise RestoreError("Die Datei kann nicht umgewandelt werden: " + "; ".join(info["problems"][:3]))
    if info["unsupported"]:
        found = ", ".join(f"{name} ({n} Schichten)" for name, n in info["unsupported"].items())
        raise RestoreError(f"Diese Quantisierungsformate werden noch nicht unterstuetzt: {found}. Unterstuetzt sind INT8 (mit und ohne ConvRot) und INT6 (W6A8).")
    layers, header, data_start = info["layers"], info["header"], info["data_start"]
    if not layers:
        raise RestoreError("In der Datei wurden keine quantisierten Schichten gefunden.")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        log("Hinweis: Keine CUDA-Grafikkarte gefunden, es wird auf dem Prozessor gerechnet.")
        device = "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False     # maximum accuracy: no TF32 shortcuts
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    skipped = info["skipped"]
    order = sorted((k for k in header if k != "__metadata__" and k not in skipped),
                   key=lambda k: int(header[k]["data_offsets"][0]))

    # Output header: same tensors, same order, quantized weights replaced by full precision ones.
    out_header, offset = {}, 0
    for key in order:
        base = key[: -len(".weight")] if key.endswith(".weight") else None
        if base in layers:
            shape = [layers[base]["rows"], layers[base]["cols"]]
            nbytes = shape[0] * shape[1] * size
            out_header[key] = {"dtype": st_name, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        else:
            nbytes = _nbytes(header[key])
            out_header[key] = {"dtype": header[key]["dtype"], "shape": header[key]["shape"],
                               "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    meta = {k: v for k, v in info["metadata"].items() if k != QUANT_META_KEY}
    meta.setdefault("format", "pt")
    out_header["__metadata__"] = meta
    blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    data_total = offset

    dropped = sum(1 for k in skipped if k.endswith(DROPPED_SUFFIXES))
    log(f"Quantisierte Schichten: {len(layers)} | Tensoren gesamt: {len(order)} | Ausgabe: {dtype} | Rechengeraet: {device}")
    if dropped:
        log(f"Nicht uebernommen: {dropped} Aufmerksamkeits-Konfigurationen (comfy_attention.config), das sind keine Gewichte.")
    partial = output_path + ".partial"
    stats = {"layers": len(layers), "copied": 0, "mismatches": 0, "checked": 0}
    started = time.time()
    try:
        with open(input_path, "rb") as src, open(partial, "wb") as dst:
            dst.write(struct.pack("<Q", len(blob)))
            dst.write(blob)
            total = len(order)
            for index, key in enumerate(order, 1):
                base = key[: -len(".weight")] if key.endswith(".weight") else None
                if base in layers:
                    layer = layers[base]
                    q = torch.frombuffer(bytearray(_read_range(src, data_start, header[key])), dtype=torch.int8).reshape(header[key]["shape"])
                    scale = torch.frombuffer(bytearray(_read_range(src, data_start, header[layer["scale_key"]])), dtype=torch.float32).reshape(layer["scale_shape"])
                    rel = None
                    if layer["rel_key"]:
                        rel = torch.frombuffer(bytearray(_read_range(src, data_start, header[layer["rel_key"]])), dtype=torch.float8_e4m3fn).reshape(header[layer["rel_key"]]["shape"])
                    try:
                        w, bad = convert_layer(layer, q, scale, rel, device, verify)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        # Not enough free VRAM (ComfyUI may be using it) or no usable GPU: do this layer on the CPU.
                        if device == "cpu":
                            raise
                        torch.cuda.empty_cache()
                        if not isinstance(e, torch.cuda.OutOfMemoryError):
                            log(f"Hinweis: Die Grafikkarte ist nicht nutzbar ({str(e).splitlines()[0][:100]}), es wird auf dem Prozessor weitergerechnet.")
                            device = "cpu"
                        w, bad = convert_layer(layer, q, scale, rel, "cpu", verify)
                    if verify:
                        stats["checked"] += 1
                        stats["mismatches"] += bad
                        if bad:
                            log(f"WARNUNG: {base}: {bad} Werte stimmen nach der Rueckrechnung nicht ueberein.")
                    dst.write(_to_bytes(w.to(out_dtype)))
                    log(f"({index}/{total}) Entpackt ({'INT6' if layer['kind'] == 'w6a8' else 'INT8'} -> {dtype}): {key}")
                else:
                    src.seek(data_start + int(header[key]["data_offsets"][0]))
                    _copy(src, dst, _nbytes(header[key]))
                    stats["copied"] += 1
                    if stats["copied"] % 25 == 0 or index == total:
                        log(f"({index}/{total}) Unveraendert kopiert: {key}")
        expected = 8 + len(blob) + data_total
        if os.path.getsize(partial) != expected:
            raise RestoreError(f"Ausgabedatei hat die falsche Groesse ({os.path.getsize(partial)} statt {expected} Bytes).")
        os.replace(partial, output_path)
    finally:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
    stats["seconds"] = time.time() - started
    stats["bytes"] = 8 + len(blob) + data_total
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description="Wandelt quantisierte Modelle (INT8 ConvRot, INT6 W6A8) zurueck in bf16/fp16/fp32.")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--dtype", default="bf16", choices=sorted(OUTPUT_DTYPES))
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--no-verify", action="store_true", help="skip the self-check (re-quantizing every layer)")
    args = p.parse_args(argv)
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("FEHLER: Ausgabe darf nicht die Eingabedatei sein.", flush=True)
        return 2
    try:
        stats = restore(args.input, args.output, args.dtype, args.device, not args.no_verify, log=lambda m: print(m, flush=True))
    except RestoreError as e:
        print(f"FEHLER: {e}", flush=True)
        return 2
    if stats["checked"]:
        if stats["mismatches"]:
            print(f"Selbstpruefung: {stats['mismatches']} abweichende Werte in {stats['checked']} Schichten.", flush=True)
        else:
            print(f"Selbstpruefung: alle {stats['checked']} Schichten stimmen exakt ueberein (erneutes Quantisieren ergibt dieselben Werte).", flush=True)
    print(f"Fertig: {stats['layers']} Schichten entpackt, {stats['copied']} Tensoren unveraendert, {stats['bytes'] / 1024 ** 3:.2f} GiB in {stats['seconds']:.0f} s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
