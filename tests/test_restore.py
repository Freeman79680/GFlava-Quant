"""Tests for restore_bf16.py (INT8 -> bf16) and its server endpoints.

The conversion needs torch (and numpy); without them the tests that run it are skipped. Everything else
(scanning, naming, the endpoints' error handling) runs without torch.
"""
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import restore_bf16 as rb  # noqa: E402
import quant_server as q  # noqa: E402

try:
    import torch
    import numpy  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def write_safetensors(path, tensors, metadata=None):
    """Independent minimal writer: tensors = {name: (dtype, shape, raw bytes)}."""
    header, offset = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    if metadata is not None:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for _, _, raw in tensors.values():
            f.write(raw)


def read_back(path):
    """Reads a .safetensors file with plain Python: {name: (dtype, shape, bytes)}, metadata."""
    header, start = rb.read_header(path)
    meta = header.pop("__metadata__", None)
    out = {}
    with open(path, "rb") as f:
        for name, e in header.items():
            f.seek(start + e["data_offsets"][0])
            out[name] = (e["dtype"], e["shape"], f.read(e["data_offsets"][1] - e["data_offsets"][0]))
    return out, meta


def marker(**fields):
    return ("U8", [0], json.dumps(fields).encode())


def int8_layer(weight, group, rotate=True):
    """What ctq does for INT8 row-wise (with ConvRot): rotate, one scale per row, round to int8."""
    w = weight.float()
    if rotate:
        h = rb.hadamard(group, "cpu")
        w = (w.reshape(w.shape[0], -1, group) @ h.T).reshape(w.shape)
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    qdata = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    return qdata, scale


def to_raw(t):
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


@unittest.skipUnless(HAVE_TORCH, "torch/numpy not installed")
class Restore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        torch.manual_seed(0)
        self.w_big = (torch.randn(64, 512) * 0.05).to(torch.bfloat16)      # ConvRot, group 256
        self.w_small = (torch.randn(32, 192) * 0.05).to(torch.bfloat16)    # ConvRot, group 64 (192 % 256 != 0)
        self.w_plain = (torch.randn(16, 40) * 0.05).to(torch.bfloat16)     # INT8 without ConvRot
        self.bias = (torch.randn(64) * 0.01).to(torch.bfloat16)
        self.norm = torch.ones(512, dtype=torch.float32)
        tensors = {}
        for name, w, group, rotate in (("blocks.0.big", self.w_big, 256, True), ("blocks.0.small", self.w_small, 64, True),
                                       ("blocks.0.plain", self.w_plain, None, False)):
            qd, sc = int8_layer(w, group, rotate)
            tensors[name + ".weight"] = ("I8", qd.shape, to_raw(qd))
            tensors[name + ".weight_scale"] = ("F32", sc.shape, to_raw(sc))
            tensors[name + ".comfy_quant"] = marker(format="int8_tensorwise", convrot=rotate, **({"convrot_groupsize": group} if rotate else {}))
        tensors["blocks.0.big.bias"] = ("BF16", self.bias.shape, to_raw(self.bias))
        tensors["final_norm.weight"] = ("F32", self.norm.shape, to_raw(self.norm))
        self.src = self.tmp / "model_int8_convrot.safetensors"
        write_safetensors(self.src, tensors, {"format": "pt", "_quantization_metadata": "{}", "note": "keep me"})
        self.out = self.tmp / "restored.safetensors"

    def rel(self, got_raw, original):
        got = torch.frombuffer(bytearray(got_raw), dtype=torch.bfloat16).reshape(original.shape).float()
        return ((got - original.float()).norm() / original.float().norm()).item()

    def test_scan_finds_all_three_kinds(self):
        info = rb.scan(str(self.src))
        self.assertEqual(sorted(info["layers"]), ["blocks.0.big", "blocks.0.plain", "blocks.0.small"])
        self.assertEqual(info["layers"]["blocks.0.small"]["group"], 64)
        self.assertFalse(info["layers"]["blocks.0.plain"]["convrot"])
        summary = rb.summarize(info)
        self.assertTrue(summary["supported"])
        self.assertEqual(summary["quantized_layers"], 3)
        self.assertLess(summary["output_bytes"]["bf16"], summary["output_bytes"]["fp32"])
        self.assertGreater(summary["output_bytes"]["bf16"], summary["input_bytes"])

    def test_restores_values_names_and_untouched_tensors(self):
        stats = rb.restore(str(self.src), str(self.out), "bf16", "cpu", True, log=lambda m: None)
        self.assertEqual((stats["layers"], stats["copied"], stats["mismatches"], stats["checked"]), (3, 2, 0, 3))
        tensors, meta = read_back(self.out)
        self.assertEqual(sorted(tensors), sorted(["blocks.0.big.weight", "blocks.0.small.weight", "blocks.0.plain.weight",
                                                  "blocks.0.big.bias", "final_norm.weight"]))
        self.assertEqual({k: v for k, v in meta.items()}, {"format": "pt", "note": "keep me"})   # quantization metadata dropped
        for name, original in (("big", self.w_big), ("small", self.w_small), ("plain", self.w_plain)):
            dtype, shape, raw = tensors[f"blocks.0.{name}.weight"]
            self.assertEqual((dtype, shape), ("BF16", list(original.shape)))
            self.assertLess(self.rel(raw, original), 0.015, name)      # INT8 rounding only, ~1 %
        # not quantized -> byte for byte the same
        self.assertEqual(tensors["blocks.0.big.bias"][2], to_raw(self.bias))
        self.assertEqual(tensors["final_norm.weight"], ("F32", [512], to_raw(self.norm)))
        self.assertFalse(Path(str(self.out) + ".partial").exists())

    def test_matches_an_independent_calculation_bit_for_bit(self):
        rb.restore(str(self.src), str(self.out), "bf16", "cpu", False, log=lambda m: None)
        tensors, _ = read_back(self.out)
        src_tensors, _ = read_back(self.src)
        qd = torch.frombuffer(bytearray(src_tensors["blocks.0.big.weight"][2]), dtype=torch.int8).reshape(64, 512)
        sc = torch.frombuffer(bytearray(src_tensors["blocks.0.big.weight_scale"][2]), dtype=torch.float32).reshape(64, 1)
        h = rb.hadamard(256, "cpu")
        expected = ((qd.float() * sc).reshape(64, 2, 256) @ h.T).reshape(64, 512).to(torch.bfloat16)
        self.assertEqual(tensors["blocks.0.big.weight"][2], to_raw(expected))

    def test_graphics_card_choice_gives_the_same_values_within_one_bf16_step(self):
        # "auto" means CUDA when there is one, otherwise the CPU; "cuda" without a card falls back to the CPU.
        cpu_out, gpu_out = self.tmp / "cpu.safetensors", self.tmp / "gpu.safetensors"
        rb.restore(str(self.src), str(cpu_out), "bf16", "cpu", True, log=lambda m: None)
        messages = []
        stats = rb.restore(str(self.src), str(gpu_out), "bf16", "cuda", True, log=messages.append)
        self.assertEqual(stats["mismatches"], 0)
        if not torch.cuda.is_available():
            self.assertTrue(any("Keine CUDA-Grafikkarte" in m for m in messages), messages)
        a, _ = read_back(cpu_out)
        b, _ = read_back(gpu_out)
        for name in ("blocks.0.big.weight", "blocks.0.small.weight", "blocks.0.plain.weight"):
            x = torch.frombuffer(bytearray(a[name][2]), dtype=torch.bfloat16).float()
            y = torch.frombuffer(bytearray(b[name][2]), dtype=torch.bfloat16).float()
            self.assertTrue(torch.allclose(x, y, rtol=2 ** -7, atol=1e-6), name)   # at most one bf16 step apart

    def test_other_output_dtypes(self):
        for dtype, st in (("fp32", "F32"), ("fp16", "F16")):
            out = self.tmp / f"restored_{dtype}.safetensors"
            rb.restore(str(self.src), str(out), dtype, "cpu", False, log=lambda m: None)
            tensors, _ = read_back(out)
            self.assertEqual(tensors["blocks.0.big.weight"][0], st)
            self.assertEqual(len(tensors["blocks.0.big.weight"][2]), 64 * 512 * (4 if dtype == "fp32" else 2))
            self.assertEqual(tensors["blocks.0.big.bias"][0], "BF16")          # untouched tensors keep their dtype

    def test_output_is_a_valid_safetensors_file(self):
        try:
            from safetensors import safe_open
        except ImportError:
            self.skipTest("safetensors not installed")
        rb.restore(str(self.src), str(self.out), "bf16", "cpu", True, log=lambda m: None)
        with safe_open(str(self.out), "pt") as f:
            self.assertEqual(sorted(f.keys()), sorted(["blocks.0.big.weight", "blocks.0.small.weight", "blocks.0.plain.weight",
                                                       "blocks.0.big.bias", "final_norm.weight"]))
            self.assertEqual(f.get_tensor("blocks.0.big.weight").dtype, torch.bfloat16)
            self.assertTrue(torch.equal(f.get_tensor("blocks.0.big.bias"), self.bias))

    def test_same_result_as_comfyui_reference_if_available(self):
        try:
            from comfy_kitchen.backends.eager import quantization as ck
        except ImportError:
            self.skipTest("comfy_kitchen not installed")
        src_tensors, _ = read_back(self.src)
        qd = torch.frombuffer(bytearray(src_tensors["blocks.0.big.weight"][2]), dtype=torch.int8).reshape(64, 512)
        sc = torch.frombuffer(bytearray(src_tensors["blocks.0.big.weight_scale"][2]), dtype=torch.float32).reshape(64, 1)
        reference = ck.dequantize_int8_convrot_weight_dtype(qd, sc, 256, ck.DTYPE_TO_CODE[torch.bfloat16])
        rb.restore(str(self.src), str(self.out), "bf16", "cpu", False, log=lambda m: None)
        tensors, _ = read_back(self.out)
        self.assertEqual(tensors["blocks.0.big.weight"][2], to_raw(reference))

    def _renamed(self, rename):
        tensors, meta = read_back(self.src)
        return {rename(k): v for k, v in tensors.items()}, meta

    def test_markers_without_the_prefix_the_weights_carry(self):
        # Seen in a model from a third-party converter: weights and scales are called
        # 'model.diffusion_model.layers...' but the markers just 'layers...' (ComfyUI strips the prefix when loading).
        prefix = "model.diffusion_model."
        tensors, meta = self._renamed(lambda k: k if k.endswith(".comfy_quant") else prefix + k)
        prefixed = self.tmp / "prefixed.safetensors"
        write_safetensors(prefixed, tensors, meta)
        summary = rb.summarize(rb.scan(str(prefixed)))
        self.assertTrue(summary["supported"], summary)
        self.assertEqual(summary["quantized_layers"], 3)
        rb.restore(str(prefixed), str(self.out), "bf16", "cpu", True, log=lambda m: None)
        restored, _ = read_back(self.out)
        self.assertEqual(sorted(restored), sorted(prefix + k for k in
                                                  ["blocks.0.big.weight", "blocks.0.small.weight", "blocks.0.plain.weight",
                                                   "blocks.0.big.bias", "final_norm.weight"]))

    def test_markers_with_a_prefix_the_weights_do_not_carry(self):
        tensors, meta = self._renamed(lambda k: "model.diffusion_model." + k if k.endswith(".comfy_quant") else k)
        odd = self.tmp / "odd.safetensors"
        write_safetensors(odd, tensors, meta)
        info = rb.scan(str(odd))
        self.assertEqual(sorted(info["layers"]), ["blocks.0.big", "blocks.0.plain", "blocks.0.small"])
        self.assertEqual(info["problems"], [])

    def test_an_ambiguous_marker_name_is_a_problem_not_a_guess(self):
        tensors, meta = read_back(self.src)
        for suffix in (".weight", ".weight_scale"):
            tensors["other." + "blocks.0.big" + suffix] = tensors["blocks.0.big" + suffix]
        write_safetensors(self.tmp / "twice.safetensors", {("a." + k if k.startswith("blocks.0.big.w") else k): v
                                                           for k, v in tensors.items() if not k.startswith("other.")}
                          | {"b." + k[len("other."):]: v for k, v in tensors.items() if k.startswith("other.")}, meta)
        info = rb.scan(str(self.tmp / "twice.safetensors"))
        self.assertTrue(any("blocks.0.big" in p for p in info["problems"]), info["problems"])

    def test_unsupported_format_is_refused_before_writing(self):
        tensors, meta = read_back(self.src)
        tensors["blocks.0.big.comfy_quant"] = marker(format="float8_e4m3fn")
        bad = self.tmp / "mixed.safetensors"
        write_safetensors(bad, tensors, meta)
        with self.assertRaises(rb.RestoreError) as ctx:
            rb.restore(str(bad), str(self.out), "bf16", "cpu", True, log=lambda m: None)
        self.assertIn("float8_e4m3fn", str(ctx.exception))
        self.assertFalse(self.out.exists())

    def test_file_without_quantization_is_refused(self):
        plain = self.tmp / "plain.safetensors"
        write_safetensors(plain, {"a.weight": ("BF16", [2, 2], b"\0" * 8)})
        with self.assertRaises(rb.RestoreError):
            rb.restore(str(plain), str(self.out), "bf16", "cpu", True, log=lambda m: None)

    def test_truncated_input_leaves_no_output_and_no_partial_file(self):
        cut = self.tmp / "cut.safetensors"
        cut.write_bytes(self.src.read_bytes()[:-200])
        with self.assertRaises(Exception):
            rb.restore(str(cut), str(self.out), "bf16", "cpu", True, log=lambda m: None)
        self.assertFalse(self.out.exists())
        self.assertFalse(Path(str(self.out) + ".partial").exists())

    def test_self_check_notices_a_broken_conversion(self):
        # A scale that does not belong to the data makes re-quantizing disagree -> the check must report it.
        tensors, meta = read_back(self.src)
        dtype, shape, raw = tensors["blocks.0.plain.weight_scale"]
        tensors["blocks.0.plain.weight_scale"] = (dtype, shape, struct.pack("<f", 0.0) + raw[4:])   # zero scale
        broken = self.tmp / "broken.safetensors"
        write_safetensors(broken, tensors, meta)
        stats = rb.restore(str(broken), str(self.out), "bf16", "cpu", True, log=lambda m: None)
        self.assertGreater(stats["mismatches"], 0)


def pack_w6a8(codes):
    """Independent packer for the 6-bit storage: a nibble plane (even column = low nibble), then a plane with the
    top two bits of every code (column c: byte c // 4, bit 2 * (c % 4)). codes: list of rows of ints 0..63."""
    out = bytearray()
    for row in codes:
        k = len(row)
        low = bytearray((row[2 * i] & 0xF) | ((row[2 * i + 1] & 0xF) << 4) for i in range(k // 2))
        high = bytearray(sum(((row[4 * b + j] >> 4) & 0x3) << (2 * j) for j in range(4)) for b in range(k // 4))
        out += low + high
    return bytes(out)


def build_w6a8_file(path, rows=16, cols=256, with_marker_tensors=False, extra=None, seed=3):
    """A small W6A8 model with known contents. Returns the expected float32 weight (original basis)."""
    import random
    rng = random.Random(seed)
    group_size, convrot = 16, 256
    codes = [[rng.randint(1, 63) for _ in range(cols)] for _ in range(rows)]
    rel_choices = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]                       # all exactly representable in e4m3
    rel = [[rng.choice(rel_choices) for _ in range(cols // group_size)] for _ in range(rows)]
    channel_t = torch.tensor([rng.uniform(0.001, 0.01) for _ in range(rows)], dtype=torch.float32)   # as stored: fp32
    # expected decoding, written out the slow, obvious way
    rot = torch.zeros(rows, cols)
    for r in range(rows):
        for c in range(cols):
            level = max(-127, min(127, round((codes[r][c] - 32) * rel[r][c // group_size])))
            rot[r, c] = torch.tensor(float(level), dtype=torch.float32) * channel_t[r]
    h = rb.hadamard(convrot, "cpu")
    expected = (rot.reshape(rows, cols // convrot, convrot) @ h.T).reshape(rows, cols)
    rel_t = torch.tensor(rel, dtype=torch.float32).to(torch.float8_e4m3fn)
    description = {"format": "w6a8_int8", "group_size": group_size, "convrot_groupsize": convrot}
    tensors = {
        "layers.0.proj.qkv.weight": ("I8", [rows, cols * 3 // 4], pack_w6a8(codes)),
        "layers.0.proj.qkv.weight_s_channel": ("F32", [rows], to_raw(channel_t)),
        "layers.0.proj.qkv.weight_s_rel": ("F8_E4M3", list(rel_t.shape), to_raw(rel_t)),
        "norm.weight": ("BF16", [4], to_raw(torch.ones(4, dtype=torch.bfloat16))),
    }
    if with_marker_tensors:
        tensors["layers.0.proj.qkv.comfy_quant"] = ("U8", [0], json.dumps(description).encode())
    tensors.update(extra or {})
    metadata = {"format": "pt"}
    if not with_marker_tensors:
        metadata[rb.QUANT_META_KEY] = json.dumps({"format_version": "1.0", "layers": {"layers.0.proj.qkv": description}})
    write_safetensors(path, tensors, metadata)
    return expected


@unittest.skipUnless(HAVE_TORCH, "torch/numpy not installed")
class RestoreSixBit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = self.tmp / "model_w6a8.safetensors"
        self.out = self.tmp / "restored.safetensors"

    def restored(self, **kwargs):
        rb.restore(str(self.src), str(self.out), "bf16", "cpu", True, log=lambda m: None)
        tensors, meta = read_back(self.out)
        return tensors, meta

    def test_decodes_the_packed_codes_exactly(self):
        expected = build_w6a8_file(self.src)
        stats = rb.restore(str(self.src), str(self.out), "bf16", "cpu", True, log=lambda m: None)
        self.assertEqual((stats["layers"], stats["mismatches"]), (1, 0))
        tensors, meta = read_back(self.out)
        self.assertEqual(sorted(tensors), ["layers.0.proj.qkv.weight", "norm.weight"])   # scales are gone
        dtype, shape, raw = tensors["layers.0.proj.qkv.weight"]
        self.assertEqual((dtype, shape), ("BF16", [16, 256]))                              # unpacked: K, not 3K/4
        self.assertEqual(raw, to_raw(expected.to(torch.bfloat16)))
        self.assertEqual(meta, {"format": "pt"})                                            # quantization metadata dropped

    def test_description_in_the_metadata_only_or_as_tensor_gives_the_same_result(self):
        build_w6a8_file(self.src, with_marker_tensors=False)
        only_metadata, _ = self.restored()
        build_w6a8_file(self.src, with_marker_tensors=True)
        self.out.unlink()
        with_tensor, _ = self.restored()
        self.assertEqual(only_metadata, with_tensor)

    def test_scan_reports_the_format(self):
        build_w6a8_file(self.src)
        summary = rb.summarize(rb.scan(str(self.src)))
        self.assertTrue(summary["supported"], summary)
        self.assertEqual(summary["kinds"], [{"format": "w6a8", "convrot": True, "group": 256, "count": 1}])
        self.assertEqual(summary["output_bytes"]["bf16"], 16 * 256 * 2 + 8)                # weight + the norm
        self.assertLess(summary["input_bytes"], summary["output_bytes"]["bf16"])

    def test_same_result_as_comfyui_reference_if_available(self):
        try:
            from comfy_kitchen.backends.eager import w4a8_int8 as ref
        except ImportError:
            self.skipTest("comfy_kitchen not installed")
        expected = build_w6a8_file(self.src)
        tensors, _ = read_back(self.src)
        q = torch.frombuffer(bytearray(tensors["layers.0.proj.qkv.weight"][2]), dtype=torch.int8).reshape(16, 192)
        rel = torch.frombuffer(bytearray(tensors["layers.0.proj.qkv.weight_s_rel"][2]), dtype=torch.float8_e4m3fn).reshape(16, 16)
        channel = torch.frombuffer(bytearray(tensors["layers.0.proj.qkv.weight_s_channel"][2]), dtype=torch.float32)
        reference = ref.dequantize_w4a8_int8_weight(q, rel, channel, None, None, 16, 256, torch.float32)
        layer = rb.scan(str(self.src))["layers"]["layers.0.proj.qkv"]
        ours, _ = rb.dequantize_w6a8(q, rel, channel, layer, "cpu")
        self.assertTrue(torch.equal(ours, reference))
        self.assertTrue(torch.equal(ours, expected))

    def test_four_bit_weights_are_refused_clearly(self):
        build_w6a8_file(self.src)
        tensors, meta = read_back(self.src)
        dtype, shape, raw = tensors["layers.0.proj.qkv.weight"]
        tensors["layers.0.proj.qkv.weight"] = (dtype, [16, 128], raw[: 16 * 128])         # K/2 bytes per row = 4-bit storage
        write_safetensors(self.src, tensors, meta)
        info = rb.scan(str(self.src))
        self.assertEqual(info["unsupported"], {"4-Bit-Gewichte (W4A8)": 1})
        with self.assertRaises(rb.RestoreError):
            rb.restore(str(self.src), str(self.out), "bf16", "cpu", True, log=lambda m: None)

    def test_attention_configs_are_not_copied(self):
        # Quantized ComfyUI models carry a tiny per-layer attention hint; it is no weight and must not end up
        # in a plain full-precision model (a merge tool would trip over a uint8 tensor).
        extra = {"layers.0.comfy_attention.config": ("U8", [35], b'{"attention": "comfy_kitchen_int8"}' + b"\0")}
        build_w6a8_file(self.src, extra=extra)
        info = rb.scan(str(self.src))
        self.assertIn("layers.0.comfy_attention.config", info["skipped"])
        tensors, _ = self.restored()
        self.assertNotIn("layers.0.comfy_attention.config", tensors)
        self.assertEqual(sorted(tensors), ["layers.0.proj.qkv.weight", "norm.weight"])

    def test_a_row_scale_of_zero_does_not_break_the_self_check(self):
        build_w6a8_file(self.src)
        tensors, meta = read_back(self.src)
        dtype, shape, raw = tensors["layers.0.proj.qkv.weight_s_channel"]
        tensors["layers.0.proj.qkv.weight_s_channel"] = (dtype, shape, struct.pack("<f", 0.0) + raw[4:])
        write_safetensors(self.src, tensors, meta)
        stats = rb.restore(str(self.src), str(self.out), "bf16", "cpu", True, log=lambda m: None)
        self.assertEqual(stats["mismatches"], 0)


class StartedLikeComfyUIsEmbeddedPython(unittest.TestCase):
    def test_restore_module_is_found_without_the_script_folder_on_sys_path(self):
        # The embedded Python of ComfyUI portable (python_embeded) has a ._pth file; with it the folder of the
        # started script is NOT on sys.path (like python -I). The server must find restore_bf16.py by itself.
        code = ("import importlib.util; "
                f"spec = importlib.util.spec_from_file_location('quant_server', r'{ROOT / 'quant_server.py'}'); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                "print('RESTORE_OK' if m.restore_bf16 is not None else 'RESTORE_MISSING')")
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tempfile.gettempdir())
        self.assertIn("RESTORE_OK", result.stdout, result.stdout + result.stderr)


class Naming(unittest.TestCase):
    def test_suggested_output_names(self):
        s = q.suggest_restore_output
        self.assertEqual(Path(s(r"C:\m\GFlavaMOSSTurbo_int8_convrot.safetensors")).name, "GFlavaMOSSTurbo_bf16.safetensors")
        self.assertEqual(Path(s("/m/moss_image_sft_int8_convrot.safetensors", "fp16")).name, "moss_image_sft_fp16.safetensors")
        self.assertEqual(Path(s("/m/model-INT8.safetensors")).name, "model-bf16.safetensors")
        self.assertEqual(Path(s("/m/mossImageINT8INT6_mossImageINT8.safetensors")).name, "mossImageINT8INT6_mossImageINT8_bf16.safetensors")
        self.assertEqual(Path(s("/m/something.safetensors")).name, "something_bf16.safetensors")


@unittest.skipUnless(HAVE_TORCH, "torch/numpy not installed")
class Endpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "logs").mkdir()
        patcher = mock.patch.object(q, "LOG_DIR", self.tmp / "logs")
        patcher.start()
        self.addCleanup(patcher.stop)
        torch.manual_seed(1)
        w = (torch.randn(32, 256) * 0.05).to(torch.bfloat16)
        qd, sc = int8_layer(w, 256, True)
        self.src = self.tmp / "tiny_int8_convrot.safetensors"
        write_safetensors(self.src, {
            "x.weight": ("I8", qd.shape, to_raw(qd)), "x.weight_scale": ("F32", sc.shape, to_raw(sc)),
            "x.comfy_quant": marker(format="int8_tensorwise", convrot=True, convrot_groupsize=256),
            "x.bias": ("BF16", [32], to_raw(torch.zeros(32, dtype=torch.bfloat16))),
        })
        self.client = q.app.test_client()

    def test_inspect(self):
        r = self.client.get("/api/restore/inspect", query_string={"path": str(self.src)})
        data = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(data["supported"])
        self.assertEqual(data["quantized_layers"], 1)
        self.assertEqual(data["kinds"], [{"format": "int8", "convrot": True, "group": 256, "count": 1}])
        self.assertTrue(data["suggested_output"].endswith("tiny_bf16.safetensors"))
        self.assertEqual(self.client.get("/api/restore/inspect", query_string={"path": str(self.tmp / "nope.safetensors")}).status_code, 400)

    def test_refuses_bad_requests(self):
        post = lambda **body: self.client.post("/api/restore", json=body)
        good = {"input_path": str(self.src), "output_path": str(self.tmp / "out.safetensors")}
        self.assertEqual(post(**dict(good, dtype="int4")).status_code, 400)
        self.assertEqual(post(**dict(good, output_path=str(self.src))).status_code, 400)               # would overwrite the input
        self.assertEqual(post(**dict(good, output_path=str(self.tmp / "out.bin"))).status_code, 400)
        self.assertEqual(post(**dict(good, output_path=str(self.tmp / "missing_dir" / "o.safetensors"))).status_code, 400)
        existing = self.tmp / "exists.safetensors"
        existing.write_bytes(b"keep")
        self.assertEqual(post(**dict(good, output_path=str(existing))).status_code, 409)               # never overwrites
        self.assertEqual(existing.read_bytes(), b"keep")
        plain = self.tmp / "plain.safetensors"
        write_safetensors(plain, {"a.weight": ("BF16", [1, 1], b"\0\0")})
        self.assertEqual(post(**dict(good, input_path=str(plain))).status_code, 400)                   # nothing quantized in it

    def test_runs_as_a_job_in_the_queue(self):
        out = self.tmp / "tiny_bf16.safetensors"
        r = self.client.post("/api/restore", json={"input_path": str(self.src), "output_path": str(out), "dtype": "bf16"})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        job_id = r.get_json()["job_id"]
        deadline = time.time() + 120
        status = {}
        while time.time() < deadline:
            status = self.client.get(f"/api/status/{job_id}").get_json()
            if status["status"] in q.FINAL_STATES:
                break
            time.sleep(0.3)
        self.assertEqual(status["status"], "done", "\n".join(status.get("log", [])[-8:]))
        self.assertEqual(status["format"], q.RESTORE_FORMAT_KEY)
        self.assertEqual(status["progress"], 100)
        self.assertTrue(out.is_file())
        tensors, _ = read_back(out)
        self.assertEqual(sorted(tensors), ["x.bias", "x.weight"])
        self.assertIn("Selbstpruefung: alle 1 Schichten stimmen exakt ueberein", "\n".join(status["log"]))


if __name__ == "__main__":
    unittest.main()
