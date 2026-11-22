"""Tests for quant_server.py. Needs only Flask (plus psutil if available); no ctq, GPU or model files.

    python -m unittest discover -s tests -v

They run on Windows, Linux and macOS. The platform-specific parts (RAM detection, process group
kill, ctq lookup on the PATH) are tested where they apply and skipped elsewhere.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import quant_server as q  # noqa: E402  (importing creates config.json/logs next to it if missing)

IS_WINDOWS = os.name == "nt"
GB = 1024 ** 3


class RamDetection(unittest.TestCase):
    MEMINFO = "MemTotal:       32768000 kB\nMemFree:         1024000 kB\nMemAvailable:   24576000 kB\n"
    VM_STAT = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               10000.\n"
        "Pages active:                            300000.\n"
        "Pages inactive:                          200000.\n"
        "Pages speculative:                         5000.\n"
        "Pages wired down:                         90000.\n"
    )

    def test_meminfo(self):
        self.assertEqual(q._parse_meminfo(self.MEMINFO), 24576000 * 1024)
        self.assertIsNone(q._parse_meminfo("MemTotal: 1 kB\n"))  # old kernels have no MemAvailable
        self.assertIsNone(q._parse_meminfo(""))

    def test_vm_stat(self):
        self.assertEqual(q._parse_vm_stat(self.VM_STAT), (10000 + 200000 + 5000) * 16384)
        self.assertEqual(q._parse_vm_stat(self.VM_STAT.replace("16384", "4096")), (10000 + 200000 + 5000) * 4096)
        self.assertIsNone(q._parse_vm_stat("Pages free: 5.\n"))  # no page size
        self.assertIsNone(q._parse_vm_stat(""))

    def test_available_ram_is_found_on_this_machine(self):
        # The reported bug: on Linux/macOS this used to be None, which forced --low-memory every time.
        ram = q.get_available_ram_bytes()
        self.assertIsNotNone(ram)
        self.assertGreater(ram, 100 * 1024 * 1024)

    def test_auto_low_memory_decision(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(b"0" * 1024)
        try:
            for available, expected in ((64 * 1024, False), (1500, True), (None, True)):
                with mock.patch.object(q, "get_available_ram_bytes", return_value=available):
                    self.assertIs(q.resolve_low_memory_flag("auto", f.name, lambda m: None), expected, available)
        finally:
            os.unlink(f.name)


def moss_image_keys(layers=2):
    """Tensor names of MOSS-Image's transformer, as listed in the published weights' index file."""
    keys = ["video_embedder.weight", "video_embedder.bias", "final_linear_video.weight", "final_norm_video.weight",
            "video_head_modulation.weight", "video_head_modulation.bias", "shared_block_modulation.weight",
            "shared_block_modulation.bias", "rope.bands", "text_embedder.proj.weight", "text_embedder.proj.bias",
            "time_embedding.0.weight", "time_embedding.0.bias", "time_embedding.2.weight", "time_embedding.2.bias"]
    for b in range(2):
        keys += [f"text_embedder.blocks.{b}.{n}" for n in
                 ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias", "ffn_up.weight", "ffn_gate.weight",
                  "ffn_down.weight", "norm1.weight", "norm2.weight", "q_norm.weight", "k_norm.weight")]
    for i in range(layers):
        keys += [f"layers.{i}.proj.{n}.weight" for n in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")]
        keys += [f"layers.{i}.{n}.weight" for n in ("norm1", "norm2", "q_norm", "k_norm")]
    return keys


class ModelDetection(unittest.TestCase):
    def test_moss_image_is_detected_with_and_without_a_key_prefix(self):
        self.assertEqual(q.detect_model_type(moss_image_keys()), "moss_image_custom")
        self.assertEqual(q.detect_model_type(["diffusion_model." + k for k in moss_image_keys()]), "moss_image_custom")

    def test_moss_image_needs_the_complete_set_of_names(self):
        keys = [k for k in moss_image_keys() if not k.startswith("shared_block_modulation")]
        self.assertNotEqual(q.detect_model_type(keys), "moss_image_custom")
        self.assertNotEqual(q.detect_model_type(["layers.0.proj.q.weight", "layers.0.proj.ffn_up_gate.weight"]),
                            "moss_image_custom")

    def test_other_models_are_not_taken_for_moss_image(self):
        qwen = ["transformer_blocks.0.img_mlp.net.0.proj.weight", "txt_in.text_norm.weight", "modulation.1.weight"]
        flux = ["double_blocks.0.img_mod.lin.weight", "double_blocks.0.txt_mod.lin.weight", "single_blocks.0.linear1.weight"]
        self.assertEqual(q.detect_model_type(qwen), "qwen21_custom")
        self.assertEqual(q.detect_model_type(flux), "flux1_custom")

    def test_moss_image_preset_protects_the_edges_and_quantizes_the_blocks(self):
        import re
        preset = q.EXCLUDE_PRESETS["moss_image_custom"]
        self.assertEqual(preset["kind"], "regex")
        self.assertFalse(preset["verified"])
        protected = [k for k in moss_image_keys(layers=42) if re.search(preset["value"], k)]
        quantized = [k for k in moss_image_keys(layers=42) if k.endswith(".weight") and k.startswith("layers.")
                     and ".proj." in k and not re.search(preset["value"], k)]
        self.assertEqual(len(quantized), 42 * 6)  # q, k, v, o, ffn_up_gate, ffn_down of every block
        for name in ("video_embedder.weight", "final_linear_video.weight", "time_embedding.0.weight",
                     "shared_block_modulation.weight", "video_head_modulation.weight",
                     "text_embedder.blocks.1.qkv.weight", "text_embedder.proj.weight"):
            self.assertIn(name, protected)
        self.assertFalse(any(k.startswith("layers.") for k in protected))


class Paths(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_clean_path(self):
        self.assertEqual(q._clean_path('  "C:\\m\\a.safetensors"  '), "C:\\m\\a.safetensors")
        self.assertEqual(os.path.normpath(q._clean_path("~/models/a.safetensors")),
                         os.path.normpath(os.path.join(os.path.expanduser("~"), "models", "a.safetensors")))
        self.assertEqual(q._clean_path(None), "")
        self.assertEqual(q._clean_path("/data/~backup/a"), "/data/~backup/a")  # only a leading ~ counts

    def test_find_safetensors_case_and_nesting(self):
        (self.tmp / "sub" / "deeper").mkdir(parents=True)
        for rel in ("a.safetensors", "sub/B.SAFETENSORS", "sub/deeper/c.safetensors", "sub/readme.txt"):
            (self.tmp / rel).write_bytes(b"x")
        names = sorted(p.name for p in q._find_safetensors(self.tmp))
        self.assertEqual(names, ["B.SAFETENSORS", "a.safetensors", "c.safetensors"])

    def test_find_safetensors_follows_symlinked_folders_without_looping(self):
        real = self.tmp / "real_models"
        real.mkdir()
        (real / "m.safetensors").write_bytes(b"x")
        root = self.tmp / "root"
        root.mkdir()
        try:
            os.symlink(real, root / "linked", target_is_directory=True)
            os.symlink(root, real / "back_to_root", target_is_directory=True)  # cycle
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available here")
        found = q._find_safetensors(root)
        self.assertEqual([p.name for p in found], ["m.safetensors"])

    def test_scan_models_reports_the_modification_time(self):
        model = self.tmp / "m.safetensors"
        model.write_bytes(b"x")
        os.utime(model, (1_700_000_000, 1_700_000_000))
        roots = [{"key": "k", "label": "L", "path": str(self.tmp)}]
        with mock.patch.dict(q.CONFIG, {"model_roots": roots}):
            entries = q.scan_models()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["mtime"], 1_700_000_000)

    def test_guess_comfyui_root_layouts(self):
        # Windows portable: python_embeded next to ComfyUI
        (self.tmp / "portable" / "ComfyUI").mkdir(parents=True)
        self.assertEqual(q._guess_comfyui_root(self.tmp / "portable" / "python_embeded" / "python.exe"),
                         str(self.tmp / "portable" / "ComfyUI"))
        # Linux/macOS: venv inside the ComfyUI folder
        inside = self.tmp / "inside" / "ComfyUI"
        (inside / "custom_nodes").mkdir(parents=True)
        (inside / "main.py").write_text("")
        self.assertEqual(q._guess_comfyui_root(inside / "venv" / "bin" / "python"), str(inside))
        # Linux/macOS: venv next to the ComfyUI folder
        (self.tmp / "beside" / "ComfyUI" / "custom_nodes").mkdir(parents=True)
        self.assertEqual(q._guess_comfyui_root(self.tmp / "beside" / "venv" / "bin" / "python"),
                         str(self.tmp / "beside" / "ComfyUI"))
        # nothing recognizable
        (self.tmp / "none" / "venv" / "bin").mkdir(parents=True)
        self.assertEqual(q._guess_comfyui_root(self.tmp / "none" / "venv" / "bin" / "python"), "")

    @unittest.skipIf(IS_WINDOWS, "PATH lookup of ctq is for Linux/macOS")
    def test_ctq_found_on_path(self):
        bindir = self.tmp / "bin"
        bindir.mkdir()
        ctq = bindir / "ctq"
        ctq.write_text("#!/bin/sh\nexit 0\n")
        ctq.chmod(0o755)
        with mock.patch.dict(os.environ, {"PATH": str(bindir) + os.pathsep + os.environ.get("PATH", "")}):
            self.assertEqual(q.find_ctq_executable(None), str(ctq))
        ctq.chmod(0o644)  # not executable -> must not be picked
        with mock.patch.dict(os.environ, {"PATH": str(bindir)}), mock.patch.object(Path, "home", return_value=self.tmp):
            self.assertNotEqual(q.find_ctq_executable(None), str(ctq))


def make_fake_ctq(directory):
    """A launcher that runs tests/fake_ctq.py with the current Python, on every OS."""
    script = ROOT / "tests" / "fake_ctq.py"
    if IS_WINDOWS:
        launcher = Path(directory) / "ctq.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
    else:
        launcher = Path(directory) / "ctq"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(0o755)
    return str(launcher)


def process_alive(pid):
    """True if the process exists and is not a zombie."""
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    if IS_WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def wait_for(condition, timeout=30, interval=0.2):
    end = time.time() + timeout
    while time.time() < end:
        value = condition()
        if value:
            return value
        time.sleep(interval)
    return None


class Jobs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "logs").mkdir()
        self.pid_file = self.tmp / "child.pid"
        patches = [
            mock.patch.object(q, "LOG_DIR", self.tmp / "logs"),   # keep the real logs/ untouched
            mock.patch.object(q, "CTQ_EXE", make_fake_ctq(self.tmp)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = q.app.test_client()
        self.model = self.tmp / "model.safetensors"
        self.model.write_bytes(b"x" * 64)

    def start(self, output_name, mode):
        env = {"FAKE_MODE": mode, "FAKE_PID_FILE": str(self.pid_file)}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        resp = self.client.post("/api/quantize", json={
            "input_path": str(self.model), "output_path": str(self.tmp / output_name), "format": "fp8",
            "exclude_preset": "none", "low_memory_mode": "off",
        })
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        return resp.get_json()["job_id"]

    def status(self, job_id):
        return self.client.get(f"/api/status/{job_id}").get_json()

    def test_job_runs_to_the_end(self):
        job_id = self.start("out_done.safetensors", "finish")
        final = wait_for(lambda: self.status(job_id) if self.status(job_id)["status"] in q.FINAL_STATES else None)
        self.assertIsNotNone(final, "job did not finish")
        self.assertEqual(final["status"], "done", final)
        self.assertEqual(final["progress"], 100)
        self.assertTrue((self.tmp / "out_done.safetensors").is_file())

    def test_cancel_kills_the_whole_process_tree_and_removes_partial_output(self):
        job_id = self.start("out_cancel.safetensors", "hang")
        self.assertTrue(wait_for(self.pid_file.exists), "fake ctq never started its child")
        child_pid = int(wait_for(lambda: self.pid_file.read_text().strip() or None))
        self.assertTrue(process_alive(child_pid))
        resp = self.client.post(f"/api/cancel/{job_id}")
        self.assertEqual(resp.status_code, 200)
        final = wait_for(lambda: self.status(job_id) if self.status(job_id)["status"] in q.FINAL_STATES else None)
        self.assertIsNotNone(final, "job did not end after cancel")
        self.assertEqual(final["status"], "cancelled")
        self.assertTrue(wait_for(lambda: not process_alive(child_pid), timeout=15),
                        "the child process of ctq is still running after cancel")
        self.assertFalse((self.tmp / "out_cancel.safetensors").exists(), "partial output was not removed")


class Startup(unittest.TestCase):
    """The start must stay fast and quiet (see _ctq_installed and _quiet_env)."""

    def test_ctq_check_does_not_import_the_package(self):
        # find_spec only looks for the package; importing ctq would load PyTorch (seconds at every start).
        with mock.patch.dict(sys.modules):
            sys.modules.pop("convert_to_quant", None)
            q._ctq_installed()
            self.assertNotIn("convert_to_quant", sys.modules)

    def test_ctq_check_follows_the_installation(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(q._ctq_installed())
        with mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertTrue(q._ctq_installed())

    def test_quiet_env_hides_the_pynvml_warning_and_keeps_existing_filters(self):
        env = q._quiet_env({})
        self.assertIn("pynvml", env["PYTHONWARNINGS"])
        env = q._quiet_env({"PYTHONWARNINGS": "error::UserWarning"})
        self.assertTrue(env["PYTHONWARNINGS"].endswith(",error::UserWarning"))
        self.assertIn("pynvml", env["PYTHONWARNINGS"])

    def test_detects_a_running_server(self):
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            with mock.patch.object(q, "SERVER_PORT", port):
                self.assertTrue(q._server_already_running())
        with mock.patch.object(q, "SERVER_PORT", port):
            self.assertFalse(q._server_already_running())


if __name__ == "__main__":
    unittest.main()
