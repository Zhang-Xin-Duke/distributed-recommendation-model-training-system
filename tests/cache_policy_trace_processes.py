"""Small Host-reference end-to-end check of the fixed-trace policy experiment."""
import pathlib
import subprocess
import sys
import tempfile

for devices in ("0", "0,1,2,3"):
  with tempfile.TemporaryDirectory(prefix="eps-policy-trace-") as temporary:
    subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve().parents[1] / "scripts/cache_policy_sweep.py"),
        "--build", str(pathlib.Path(sys.argv[1]).resolve().parent),
        "--out", str(pathlib.Path(temporary) / "results"), "--backend", "host", "--smoke",
        "--repeats", "1", "--calibration-repeats", "1", "--devices", devices], check=True, timeout=120)
