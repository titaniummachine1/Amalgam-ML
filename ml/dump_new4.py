"""Dump 4th pull: every .dem in mge_demos + orange_demos lacking a player CSV."""
import concurrent.futures
import glob
import os
import subprocess
import time

DEM_DIRS = [os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_demos'),
            os.path.join(os.path.expanduser('~'), 'Downloads', 'orange_demos')]
OUT = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
os.makedirs(OUT, exist_ok=True)
BIN = r'C:\gitProjects\amalgam\ml\demodump_rs\target\release\demospike_rs.exe'
files = []
for d in DEM_DIRS:
    files += sorted(glob.glob(os.path.join(d, '*.dem')))


def run(f):
    out = os.path.join(OUT, os.path.basename(f) + '.csv')
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        return os.path.basename(f), 'skip', os.path.getsize(out)
    t = time.time()
    r = subprocess.run([BIN, f, out], capture_output=True, text=True)
    dt = time.time() - t
    ok = r.returncode == 0 and os.path.exists(out)
    return os.path.basename(f), ('ok %.1fs' % dt) if ok else ('FAIL ' + r.stderr[-500:]), \
        os.path.getsize(out) if ok else 0


t0 = time.time()
print('total files:', len(files), flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for f, st, n in ex.map(run, files):
        print(f, st, n, flush=True)
print('total %.1fs' % (time.time() - t0), flush=True)
