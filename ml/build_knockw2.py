import os, subprocess, sys, time
ML = r"C:\gitProjects\amalgam\ml"
LOG = r"C:\Users\Terminatort8000\AppData\Local\Temp\opencode\knockw2.log"

steps = [
    ({"OUT_SUFFIX": "B"}, ["python", "build_knock.py", "eval"]),
    ({"OUT_SUFFIX": "B"}, ["python", "build_knock.py", "train"]),
    ({"OUT_SUFFIX": "C", "ML_DB": "dataset_eye.db", "EYE_FRAME": "1"},
     ["python", "build_knock.py", "eval"]),
    ({"OUT_SUFFIX": "C", "ML_DB": "dataset_eye.db", "EYE_FRAME": "1"},
     ["python", "build_knock.py", "train"]),
    ({"OUT_SUFFIX": "B"}, ["python", "build_w2.py", "train"]),
]

with open(LOG, "a") as lf:
    for env, s in steps:
        e = dict(os.environ)
        e.update(env)
        e["PYTHONUNBUFFERED"] = "1"
        lf.write("=== %s %s @ %s ===\n" % (env, " ".join(s), time.strftime("%H:%M:%S")))
        lf.flush()
        t0 = time.time()
        r = subprocess.run(s, cwd=ML, env=e, capture_output=True, text=True)
        lf.write(r.stdout[-2500:] + r.stderr[-2500:])
        lf.write("rc=%d %.1f min\n" % (r.returncode, (time.time() - t0) / 60))
        lf.flush()
        if r.returncode != 0:
            lf.write("STOPPING ON FAILURE\n")
            break
    lf.write("=== DONE @ %s ===\n" % time.strftime("%H:%M:%S"))
