"""Build all generation-C caches in dependency order. Logs to TMP/cachesC.log."""
import os
import subprocess
import sys
import time

ML = r"C:\gitProjects\amalgam\ml"
LOG = r"C:\Users\Terminatort8000\AppData\Local\Temp\opencode\cachesC.log"

env = dict(os.environ)
env.update({"ML_DB": "dataset_eye.db", "DS_SUFFIX": "C", "OUT_SUFFIX": "C",
            "EYE_FRAME": "1", "PYTHONUNBUFFERED": "1"})

steps = [
    ["python", "build_dense66.py", "eval"],
    ["python", "build_dense66.py", "train"],
    ["python", "build_threatwin.py", "eval"],
    ["python", "build_threatwin.py", "train"],
    ["python", "build_mspd.py", "eval"],
    ["python", "build_mspd.py", "train"],
    ["python", "build_dyaw.py", "eval"],
    ["python", "build_dyaw.py", "train"],
]

with open(LOG, "a") as lf:
    for s in steps:
        lf.write("=== %s @ %s ===\n" % (" ".join(s), time.strftime("%H:%M:%S")))
        lf.flush()
        t0 = time.time()
        r = subprocess.run(s, cwd=ML, env=env, capture_output=True, text=True)
        lf.write(r.stdout[-3000:])
        lf.write(r.stderr[-3000:])
        lf.write("rc=%d %.1f min\n" % (r.returncode, (time.time() - t0) / 60))
        lf.flush()
        if r.returncode != 0:
            lf.write("STOPPING ON FAILURE\n")
            break
    lf.write("=== DONE @ %s ===\n" % time.strftime("%H:%M:%S"))
