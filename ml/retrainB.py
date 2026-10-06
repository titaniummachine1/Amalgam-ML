"""Rebuild all TRAIN B caches against current keys (hermetic env).
Order: dense -> threatwin -> mspd -> knock(par) -> tsf(par) -> w2."""
import os
import subprocess
import sys
import time

ML = r"C:\gitProjects\amalgam\ml"
TMP = r"C:\Users\Terminatort8000\AppData\Local\Temp\opencode"
LOG = os.path.join(TMP, "retrainB.log")


def base_env(**kw):
    e = {"PATH": os.environ["PATH"],
         "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
         "TEMP": os.environ.get("TEMP", r"C:\Temp"),
         "TMPDIR": os.environ.get("TMPDIR", r"C:\Temp"),
         "USERPROFILE": os.environ["USERPROFILE"],
         "HOMEDRIVE": os.environ.get("HOMEDRIVE", "C:"),
         "HOMEPATH": os.environ.get("HOMEPATH", r"\Users\Terminatort8000"),
         "PYTHONUNBUFFERED": "1", "PYTHONPATH": "", "PYTHONHOME": "",
         "ML_DB": "dataset_all.db", "DS_SUFFIX": "B", "OUT_SUFFIX": "B",
         "EYE_FRAME": "0"}
    e.update(kw)
    return e


STEPS = [
    ("denseB train", ["python", "build_dense66.py", "train"], {}),
    ("threatB train", ["python", "build_threatwin.py", "train"], {}),
    ("mspdB train", ["python", "build_mspd.py", "train"], {}),
    ("knockB train", ["python", "build_knockpar.py", "train", "B", "6"], {}),
    ("tsfB train", ["python", "build_tsfpar.py", "train", "B", "6"], {}),
    ("w2 train", ["python", "build_w2.py", "train"], {}),
]

with open(LOG, "a") as lf:
    for name, cmd, kw in STEPS:
        lf.write("=== %s @ %s ===\n" % (name, time.strftime("%H:%M:%S")))
        lf.flush()
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ML, env=base_env(**kw),
                           capture_output=True, text=True)
        lf.write(r.stdout[-2500:] + r.stderr[-2500:])
        lf.write("rc=%d %.1f min\n" % (r.returncode, (time.time() - t0) / 60))
        lf.flush()
        if r.returncode != 0:
            lf.write("STOPPING ON FAILURE\n")
            break
    lf.write("=== DONE @ %s ===\n" % time.strftime("%H:%M:%S"))
