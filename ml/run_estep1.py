"""E data pipeline step 1 (detached): re-dump (loadout cols) -> fresh eye DB.
Sequential in one process so ordering is guaranteed. Logs to step1 log."""
import os, subprocess, sys, time

ML = r'C:\gitProjects\amalgam\ml'
PY = r'C:\Users\Terminatort8000\AppData\Local\Python\pythoncore-3.14-64\python.exe'
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')

# fresh DB rebuild: archive current (has wcls, lacks w0-2)
db = os.path.join(DS, 'dataset_eye.db')
bak = os.path.join(DS, 'dataset_eye.wcls.db')
if os.path.exists(db):
    if os.path.exists(bak):
        os.remove(bak)
    os.rename(db, bak)
    print('archived eye DB to dataset_eye.wcls.db', flush=True)

for script in ['dump_redump.py', 'build_db_eye.py']:
    t0 = time.time()
    print('=== %s ===' % script, flush=True)
    r = subprocess.run([PY, os.path.join(ML, script)])
    print('=== %s exit=%d %.1fs ===' % (script, r.returncode, time.time() - t0), flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
print('STEP1 COMPLETE', flush=True)
