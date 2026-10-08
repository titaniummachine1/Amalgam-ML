"""Force re-dump ALL demos with the current extractor (wcls column, 2026-10-08).
Unlike dump_all.py: fresh binary path, no skip-if-exists (wcls backfill),
covers mge_demos + orange_demos. Overwrites base CSVs in place;
.proj/.hurt derivatives are regenerated downstream, not here."""
import os, subprocess, concurrent.futures, glob, time
DEM1 = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_demos')
DEM2 = os.path.join(os.path.expanduser('~'), 'Downloads', 'orange_demos')
EXTRA = [os.path.join(os.path.expanduser('~'), 'Downloads',
    'tfmge_2026-09-29__00-02_mge_training_v8_beta4b.dem')]
OUT = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
os.makedirs(OUT, exist_ok=True)
BIN = r'C:\gitProjects\amalgam\ml\demodump_rs\target\release\demospike_rs.exe'
files = sorted(glob.glob(os.path.join(DEM1, '*.dem')))
if os.path.isdir(DEM2):
    files += sorted(glob.glob(os.path.join(DEM2, '*.dem')))
files += [f for f in EXTRA if os.path.exists(f)]
print('demos:', len(files), flush=True)
def run(f):
    out = os.path.join(OUT, os.path.basename(f) + '.csv')
    t = time.time()
    r = subprocess.run([BIN, f, out], capture_output=True, text=True)
    dt = time.time() - t
    ok = r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 10000
    return f, ('ok %.1fs' % dt) if ok else ('FAIL ' + (r.stderr or '')[-300:]), os.path.getsize(out) if ok else 0
t0 = time.time()
nfail = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for f, st, n in ex.map(run, files):
        print(os.path.basename(f), st, n, flush=True)
        if st.startswith('FAIL'):
            nfail += 1
print('total %.1fs fails=%d' % (time.time() - t0, nfail), flush=True)
