import os, subprocess, concurrent.futures, glob, time
DEM = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_demos')
EXTRA = [os.path.join(os.path.expanduser('~'), 'Downloads',
    'tfmge_2026-09-29__00-02_mge_training_v8_beta4b.dem')]
OUT = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
os.makedirs(OUT, exist_ok=True)
BIN = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode\demospike_rs\target\release\demospike_rs.exe'
files = sorted(glob.glob(os.path.join(DEM, '*.dem'))) + [f for f in EXTRA if os.path.exists(f)]
def run(f):
    out = os.path.join(OUT, os.path.basename(f) + '.csv')
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        return f, 'skip', os.path.getsize(out)
    t = time.time()
    r = subprocess.run([BIN, f, out], capture_output=True, text=True)
    dt = time.time() - t
    ok = r.returncode == 0 and os.path.exists(out)
    return f, ('ok %.1fs' % dt) if ok else ('FAIL ' + r.stderr[-500:]), os.path.getsize(out) if ok else 0
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for f, st, n in ex.map(run, files):
        print(os.path.basename(f), st, n)
print('total %.1fs' % (time.time() - t0))
