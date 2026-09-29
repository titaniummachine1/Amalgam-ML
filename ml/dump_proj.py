import os, subprocess, concurrent.futures, glob, time
DEM = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_demos')
CSV = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
EXTRA = os.path.join(os.path.expanduser('~'), 'Downloads',
    'tfmge_2026-09-29__00-02_mge_training_v8_beta4b.dem')
BIN = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode\demospike_rs\target\release\demospike_rs.exe'
pairs = []
for d in sorted(glob.glob(os.path.join(DEM, '*.dem'))) + [EXTRA]:
    c = os.path.join(CSV, os.path.basename(d) + '.csv')
    if os.path.exists(c + '.proj.csv') and os.path.getsize(c + '.proj.csv') > 1000:
        continue
    pairs.append((d, c))
print('to dump:', len(pairs))
def run(p):
    d, c = p
    t = time.time()
    r = subprocess.run([BIN, d, c], capture_output=True)
    return os.path.basename(d), r.returncode, '%.1fs' % (time.time() - t)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for b, rc, dt in ex.map(run, pairs):
        print(b, rc, dt)
