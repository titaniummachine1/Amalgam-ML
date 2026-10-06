"""Parallel driver for build_knock: shard demos across workers, merge after.
Usage: python ml/build_knockpar.py <tag> [suf] [workers]
Merges shards deterministically (row indices stable); prints combined K_fit inputs.
"""
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
                'EYE_FRAME': os.environ.get('EYE_FRAME', '0'),
                'KNOCK_DIDS': ','.join(str(x) for x in ch),
                'KNOCK_SHARD': os.path.join(TMP, 'knock_%s%s_%d.npz' % (tag, suf, i)),
                'PYTHONUNBUFFERED': '1'})
    log = os.path.join(TMP, 'knock_%s%s_%d.log' % (tag, suf, i))
    lf = open(log, 'w')
    procs.append((Popen([sys.executable, os.path.join(ML, 'build_knock.py'), tag],
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
# merge
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
files = sorted(glob.glob(os.path.join(TMP, 'knock_%s%s_*.npz' % (tag, suf))))
n = int((k[:, 3] == 3).sum())
KB = np.zeros((n, 66, 4), dtype=np.float32)
KBW = np.zeros((n, 66, 3), dtype=np.float32)
seen = np.zeros(n, dtype=bool)
for f in files:
    z = np.load(f)
    if len(z['IDX']) == 0:
        continue
    KB[z['IDX']] = z['KB']
    KBW[z['IDX']] = z['KBW']
    assert not seen[z['IDX']].any(), 'overlap in shards!'
    seen[z['IDX']] = True
print('merged %d/%d rows (%.4f coverage)' % (seen.sum(), n, seen.mean()), flush=True)
assert seen.all(), 'MISSING ROWS'
np.savez_compressed(os.path.join(DS, tag + suf + '_knock.npz'), KB=KB, KBW=KBW)
has = KB[:, :, 3].sum(axis=1) > 0
print('%s%s rows w/ future hurt %.4f  w/ directed %.4f' % (
    tag, suf, has.mean(),
    float((np.abs(KB[:, :, 0:2]).sum(axis=(1, 2)) > 0).mean())), flush=True)
