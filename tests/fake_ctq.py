"""A stand-in for ctq used by the tests. It takes the same arguments (-i/-o/...) but only pretends to work.

FAKE_MODE=finish   print progress lines, write the output file and exit 0
FAKE_MODE=hang     write a partial output file, start a child process, print progress and wait
                   (used to test that "Cancel" ends the process together with its child)
FAKE_PID_FILE      the child's PID is written here, so the test can check that it is gone
"""
import os
import subprocess
import sys
import time

args = sys.argv[1:]
output = args[args.index("-o") + 1] if "-o" in args else None
mode = os.environ.get("FAKE_MODE", "finish")

if mode == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    if output:
        with open(output, "wb") as f:
            f.write(b"partial")
    with open(os.environ["FAKE_PID_FILE"], "w") as f:
        f.write(str(child.pid))
    print("(1/10) Processing (INT8): fake.weight", flush=True)
    time.sleep(120)
else:
    for i in (1, 2):
        print(f"({i}/2) Processing (INT8): fake{i}.weight", flush=True)
    if output:
        with open(output, "wb") as f:
            f.write(b"quantized")
