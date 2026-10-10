"""Dump next N missing demos (MGE then orange). Resumable. Usage: dump_batch.py [N=6]"""
import os
import subprocess
import sys
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
U = os.path.expanduser('~/Downloads')
DIRS = [U + '/mge_demos', U + '/orange_demos']
CSV = U + '/mge_csv'
BIN = r'C:\gitProjects\amalgam\ml\demodump_rs\target\release\demospike_rs.exe'
missing = []
for d in DIRS:
    for f in sorted(os.listdir(d)):
        if not f.endswith('.dem'):
            continue
        o = os.path.join(CSV, f + '.csv')
        if not (os.path.exists(o) and os.path.getsize(o) > 10000):
            missing.append((d, f))
print('missing total: %d' % len(missing), flush=True)
only = sys.argv[2] if len(sys.argv) > 2 else None
if only:
    missing = [m for m in missing if m[1] == only]
for d, f in missing[:N]:
    t = time.time()
    r = subprocess.run([BIN, os.path.join(d, f), os.path.join(CSV, f + '.csv')],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    print('%s %s %.0fs' % (f, 'OK' if ok else 'FAIL ' + r.stderr[-200:], time.time() - t), flush=True)
