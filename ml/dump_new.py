import os, subprocess, concurrent.futures, time
DEM_MGE = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_demos')
DEM_OR = os.path.join(os.path.expanduser('~'), 'Downloads', 'orange_demos')
CSV = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
BIN = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode\demospike_rs\target\release\demospike_rs.exe'
import sys
sys.path.insert(0, r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode')
import didreg
reg = didreg.load()
# dump every registered demo lacking a player CSV
pairs = []
DEM_DIRS = [DEM_MGE, DEM_OR, os.path.join(os.path.expanduser('~'), 'Downloads')]
for csvname in reg:
    dem = csvname[:-4]  # strip .csv
    src = None
    for d in DEM_DIRS:
        p = os.path.join(d, dem)
        if os.path.exists(p):
            src = p
            break
    if src is None:
        print('MISSING DEMO FILE:', dem)
        continue
    dst = os.path.join(CSV, csvname)
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        continue
    pairs.append((src, dst))
print('to dump:', len(pairs))
def run(p):
    src, dst = p
    t = time.time()
    r = subprocess.run([BIN, src, dst], capture_output=True)
    dt = time.time() - t
    ok = r.returncode == 0 and os.path.exists(dst)
    return os.path.basename(src), ok, '%.1fs' % dt, (os.path.getsize(dst) if ok else 0)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for b, ok, dt, n in ex.map(run, pairs):
        print(b, 'OK' if ok else 'FAIL', dt, n)
