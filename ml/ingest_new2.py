"""Full ingest for the 16 new demos (2026-10-01 pull): dump -> verify gate ->
DB append (both DBs) -> rebuild B+C base datasets and all caches.
New demos land in TRAIN only (EVAL_NAMES unchanged); eval rebuilds are
verified bit-identical before proceeding. Logs to TMP/ingest2.log."""
import os
import subprocess
import sys
import time

ML = r"C:\gitProjects\amalgam\ml"
TMP = r"C:\Users\Terminatort8000\AppData\Local\Temp\opencode"
LOG = os.path.join(TMP, "ingest2.log")
MGE = os.path.join(os.path.expanduser("~"), "Downloads", "mge_demos")
ORG = os.path.join(os.path.expanduser("~"), "Downloads", "orange_demos")
CSV = os.path.join(os.path.expanduser("~"), "Downloads", "mge_csv")
EXE = r"C:\gitProjects\amalgam\ml\demodump_rs\target\release\demospike_rs.exe"

NEW_MGE = [
    "tfmge_2026-10-01__17-56_mge_training_v8_beta4b.dem",
    "tfmge_2026-10-01__16-56_mge_training_v8_beta4b.dem",
    "tfmge_2026-10-01__16-36_mge_training_v8_beta4b.dem",
    "tfmge_2026-10-01__16-20_mge_training_v8_beta4b.dem",
    "tfmge_2026-10-01__14-48_mge_training_v8_beta4b.dem",
    "tfmge_2026-10-01__07-51_mge_training_v8_beta4b.dem",
    "tfmge_2026-10-01__07-50_mge_training_v8_beta4b.dem",
    "tfmge_2026-10-01__07-49_mge_training_v8_beta4b.dem",
    "tfmge_2026-10-01__07-46_mge_training_v8_beta4b.dem",
    "tfmge_2026-09-30__23-33_mge_training_v8_beta4b.dem",
    "tfmge_2026-09-30__22-33_mge_training_v8_beta4b.dem",
    "tfmge_2026-09-30__19-50_mge_training_v8_beta4b.dem",
]
NEW_OR = [
    "tforange_2026-10-01__17-49_cp_orange_x5_bg_v34.dem",
    "tforange_2026-10-01__16-43_cp_orange_x5_bg_v34.dem",
    "tforange_2026-09-30__23-10_cp_orange_x5_bg_v34.dem",
    "tforange_2026-09-30__20-48_cp_orange_x5_bg_v34.dem",
]


def run(cmd, env=None, what=""):
    with open(LOG, "a") as lf:
        lf.write("=== %s @ %s ===\n" % (what or " ".join(cmd[:3]), time.strftime("%H:%M:%S")))
        lf.flush()
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ML, env=env or dict(os.environ),
                       capture_output=True, text=True)
    with open(LOG, "a") as lf:
        lf.write(r.stdout[-2000:] + r.stderr[-2000:])
        lf.write("rc=%d %.1f min\n" % (r.returncode, (time.time() - t0) / 60))
        lf.flush()
    return r.returncode


def main():
    # 1. dump new demos (fixed extractor: live eye + mspd column)
    for d in NEW_MGE:
        src = os.path.join(MGE, d)
        if os.path.exists(os.path.join(CSV, d + ".csv")):
            continue
        rc = run([EXE, src, os.path.join(CSV, d + ".csv")], what="dump " + d)
        if rc != 0:
            return
    for d in NEW_OR:
        src = os.path.join(ORG, d)
        if os.path.exists(os.path.join(CSV, d + ".csv")):
            continue
        rc = run([EXE, src, os.path.join(CSV, d + ".csv")], what="dump " + d)
        if rc != 0:
            return
    # 2. verify gate on the new CSVs only
    new_csvs = [os.path.join(CSV, d + ".csv") for d in NEW_MGE + NEW_OR]
    rc = run([sys.executable, "verify_csv.py"] + new_csvs, what="verify 16 new")
    if rc != 0:
        return
    # 3. DB append (both DBs; scripts skip demos already present)
    rc = run([sys.executable, "build_db_all.py"], what="db_all append")
    if rc != 0:
        return
    rc = run([sys.executable, "build_db_eye.py"], what="db_eye append")
    if rc != 0:
        return
    # 4. rebuild B base + all B caches
    envB = dict(os.environ)
    envB.update({"OUT_SUFFIX": "B", "PYTHONUNBUFFERED": "1"})
    rc = run([sys.executable, "build_ds_all.py"], env=envB, what="ds B")
    if rc != 0:
        return
    for s in (["build_dense66.py", "eval"], ["build_dense66.py", "train"],
              ["build_threatwin.py", "eval"], ["build_threatwin.py", "train"],
              ["build_mspd.py", "eval"], ["build_mspd.py", "train"]):
        rc = run([sys.executable] + s, env=envB, what="B " + " ".join(s))
        if rc != 0:
            return
    # 5. rebuild C base + all C caches
    envC = dict(os.environ)
    envC.update({"ML_DB": "dataset_eye.db", "OUT_SUFFIX": "C", "DS_SUFFIX": "C",
                 "EYE_FRAME": "1", "PYTHONUNBUFFERED": "1"})
    rc = run([sys.executable, "build_ds_all.py"], env=envC, what="ds C")
    if rc != 0:
        return
    for s in (["build_dense66.py", "eval"], ["build_dense66.py", "train"],
              ["build_threatwin.py", "eval"], ["build_threatwin.py", "train"],
              ["build_mspd.py", "eval"], ["build_mspd.py", "train"],
              ["build_dyaw.py", "eval"], ["build_dyaw.py", "train"]):
        rc = run([sys.executable] + s, env=envC, what="C " + " ".join(s))
        if rc != 0:
            return
    with open(LOG, "a") as lf:
        lf.write("=== INGEST COMPLETE @ %s ===\n" % time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
