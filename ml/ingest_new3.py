"""Ingest the 15 new demos (Oct 5-6 pull): dump -> verify -> DB append ->
rebuild TRAIN datasets + caches (B and C). Eval frozen."""
import os
import subprocess
import sys
import time

ML = r"C:\gitProjects\amalgam\ml"
TMP = r"C:\Users\Terminatort8000\AppData\Local\Temp\opencode"
LOG = os.path.join(TMP, "ingest3.log")
MGE = os.path.join(os.path.expanduser("~"), "Downloads", "mge_demos")
ORG = os.path.join(os.path.expanduser("~"), "Downloads", "orange_demos")
CSV = os.path.join(os.path.expanduser("~"), "Downloads", "mge_csv")
EXE = r"C:\gitProjects\amalgam\ml\demodump_rs\target\release\demospike_rs.exe"

NEW = [
    (MGE, "tfmge_2026-10-06__03-09_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-06__03-06_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-06__03-05_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-05__22-53_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-05__22-08_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-05__22-05_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-05__22-01_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-05__21-35_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-05__20-35_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-05__19-31_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-05__17-55_mge_training_v8_beta4b.dem"),
    (MGE, "tfmge_2026-10-05__16-54_mge_training_v8_beta4b.dem"),
    (ORG, "tforange_2026-10-05__19-52_cp_orange_x5_bg_v34.dem"),
    (ORG, "tforange_2026-10-05__18-12_cp_orange_x5_bg_v34.dem"),
    (ORG, "tforange_2026-10-05__16-12_cp_orange_x5_bg_v34.dem"),
]


def base_env(**kw):
    e = {"PATH": os.environ["PATH"],
         "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
         "TEMP": os.environ.get("TEMP", r"C:\Temp"),
         "TMPDIR": os.environ.get("TMPDIR", r"C:\Temp"),
         "USERPROFILE": os.environ["USERPROFILE"],
         "HOMEDRIVE": os.environ.get("HOMEDRIVE", "C:"),
         "HOMEPATH": os.environ.get("HOMEPATH", r"\Users\Terminatort8000"),
         "PYTHONUNBUFFERED": "1", "PYTHONPATH": "", "PYTHONHOME": ""}
    e.update(kw)
    return e


def log(msg):
    with open(LOG, "a") as lf:
        lf.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))
        lf.flush()
    print(msg, flush=True)


with open(LOG, "a") as lf:
    lf.write("=== INGEST3 @ %s ===\n" % time.strftime("%H:%M:%S"))
# 1. dump
for ddir, d in NEW:
    out = os.path.join(CSV, d + ".csv")
    if os.path.exists(out) and os.path.getsize(out) > 100000:
        log("dump skip (exists) %s" % d)
        continue
    t0 = time.time()
    r = subprocess.run([EXE, os.path.join(ddir, d), out], cwd=ML,
                       capture_output=True, text=True)
    log("dump %s rc=%d %.1f min" % (d, r.returncode, (time.time() - t0) / 60))
    if r.returncode != 0:
        log("DUMP FAILED: " + r.stderr[-1000:])
        sys.exit(1)
# 2. verify
csvs = [os.path.join(CSV, d + ".csv") for _, d in NEW]
r = subprocess.run([sys.executable, "verify_csv.py"] + csvs, cwd=ML,
                   env=base_env(), capture_output=True, text=True)
log("verify rc=%d\n%s" % (r.returncode, r.stdout[-2000:]))
if r.returncode != 0:
    log("VERIFY FAILED"); sys.exit(1)
# 3. DB appends
for s in (["build_db_all.py"], ["build_db_eye.py"]):
    t0 = time.time()
    r = subprocess.run([sys.executable] + s, cwd=ML, env=base_env(),
                       capture_output=True, text=True)
    log("%s rc=%d %.1f min\n%s" % (" ".join(s), r.returncode,
                                   (time.time() - t0) / 60, r.stdout[-1500:]))
    if r.returncode != 0:
        log("DB FAILED"); sys.exit(1)
# 4. rebuild TRAIN datasets + caches (B then C)
for tag, extra in (("B", {}), ("C", {"ML_DB": "dataset_eye.db", "DS_SUFFIX": "C",
                                     "EYE_FRAME": "1"})):
    env = base_env(OUT_SUFFIX=tag, **extra)
    for s in (["build_ds_all.py"],
              ["build_dense66.py", "train"], ["build_threatwin.py", "train"],
              ["build_mspd.py", "train"], ["build_dyaw.py", "train"]):
        t0 = time.time()
        r = subprocess.run([sys.executable] + s, cwd=ML, env=env,
                           capture_output=True, text=True)
        log("%s %s rc=%d %.1f min\n%s" % (tag, " ".join(s), r.returncode,
                                          (time.time() - t0) / 60, r.stdout[-1500:]))
        if r.returncode != 0:
            log("BUILD FAILED"); sys.exit(1)
log("=== INGEST3 COMPLETE @ %s ===" % time.strftime("%H:%M:%S"))
