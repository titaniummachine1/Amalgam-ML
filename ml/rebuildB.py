"""Hermetic rebuild of generation B with the unified frame rule
(velocity heading when moving, eye when still; no std threshold).
Keys are frame-invariant so the row sets are stable; only feature/label/target
values change. Every step gets explicit complete env; validation gates each.
Order: ds -> dense -> threatwin -> knock(par) -> w2.
"""
import os
import subprocess
import sys
import time

ML = r"C:\gitProjects\amalgam\ml"
TMP = r"C:\Users\Terminatort8000\AppData\Local\Temp\opencode"
LOG = os.path.join(TMP, "rebuildB.log")


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
    ("ds B", ["python", "build_ds_all.py"], {}),
    ("denseB eval", ["python", "build_dense66.py", "eval"], {}),
    ("denseB train", ["python", "build_dense66.py", "train"], {}),
    ("threatB eval", ["python", "build_threatwin.py", "eval"], {}),
    ("threatB train", ["python", "build_threatwin.py", "train"], {}),
    ("knockB eval", ["python", "build_knockpar.py", "eval", "B", "6"], {}),
    ("knockB train", ["python", "build_knockpar.py", "train", "B", "6"], {}),
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
