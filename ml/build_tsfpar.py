"""Parallel driver for build_tsf (mirrors build_knockpar)."""
import glob
import os
import sys
import time
from subprocess import Popen

ML = os.path.dirname(os.path.abspath(__file__))
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
tag = sys.argv[1] if len(sys.argv) > 1 else 'eval'
suf = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('OUT_SUFFIX', 'B')
nw = int(sys.argv[3]) if len(sys.argv) > 3 else 6

import numpy as np
sys.path.insert(0, ML)
k = np.load(os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds',
                         tag + suf + 'k.npz'))['K']
dids = sorted(set(int(x) for x in k[np.where(k[:, 3] == 3)[0]][:, 0]))
print('%s%s: %d soldier rows across %d demos, %d workers' % (tag, suf, (k[:, 3] == 3).sum(), len(dids), nw), flush=True)
chunks = [dids[i::nw] for i in range(nw)]
procs = []
for i, ch in enumerate(chunks):
    if not ch:
        continue
    env = dict(os.environ)
    env.update({'OUT_SUFFIX': suf,
                'ML_DB': os.environ.get('ML_DB', 'dataset_all.db'),
                'TSF_DIDS': ','.join(str(x) for x in ch),
                'TSF_SHARD': os.path.join(TMP, 'tsf_%s%s_%d.npz' % (tag, suf, i)),
                'PYTHONUNBUFFERED': '1'})
    log = os.path.join(TMP, 'tsf_%s%s_%d.log' % (tag, suf, i))
    lf = open(log, 'w')
    procs.append((Popen([sys.executable, os.path.join(ML, 'build_tsf.py'), tag],
                        cwd=ML, env=env, stdout=lf, stderr=lf), lf, log))
    print('  worker %d: %d demos' % (i, len(ch)), flush=True)
rc = 0
for p, lf, log in procs:
    r = p.wait()
    lf.close()
    print('  worker rc=%d (%s)' % (r, log), flush=True)
    rc = rc or r
if rc != 0:
    print('WORKER FAILURE', flush=True)
    sys.exit(1)
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
files = sorted(glob.glob(os.path.join(TMP, 'tsf_%s%s_*.npz' % (tag, suf))))
n = int((k[:, 3] == 3).sum())
TSF = np.zeros((n, 66, 6), dtype=np.float32)
seen = np.zeros(n, dtype=bool)
for f in files:
    z = np.load(f)
    if len(z['IDX']) == 0:
        continue
    TSF[z['IDX']] = z['TSF']
    assert not seen[z['IDX']].any(), 'overlap!'
    seen[z['IDX']] = True
print('merged %d/%d rows (%.4f)' % (seen.sum(), n, seen.mean()), flush=True)
assert seen.all(), 'MISSING ROWS'
np.savez_compressed(os.path.join(DS, tag + suf + '_tsf.npz'), TSF=TSF)
print('%s%s TSF means: %s' % (tag, suf, np.round(TSF[:, 65, :].mean(axis=0), 4)), flush=True)
