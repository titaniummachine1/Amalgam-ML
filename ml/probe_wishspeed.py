"""probe_wishspeed.py — what wishspeed does the demo data actually require?
For each grounded moving tick we take the inverted direction, then bisect the
wishspeed that makes ground_step reproduce the observed velocity exactly.
If the required wishspeed is a clean constant (450 / a class speed), the
candidate model can use it and recovery becomes exact.
"""
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (FL_ONGROUND, ground_step, inverse_ground, load_csv_live)

csv = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
rows = load_csv_live(csv, classes=set(range(1, 10)))
e = max(rows, key=lambda x: len(rows[x]))
rs = rows[e]
need = []
msp = []
for k in range(N):
    if k + 1 >= len(rs):
        break
    if int(rs[k + 1]['tick']) - int(rs[k]['tick']) != 1:
        continue
    try:
        v = np.array([float(rs[k]['vx']), float(rs[k]['vy']), float(rs[k]['vz'])])
        vn = np.array([float(rs[k + 1]['vx']), float(rs[k + 1]['vy']),
                       float(rs[k + 1]['vz'])])
        fl = int(float(rs[k]['flags']))
        ms = float(rs[k].get('mspd') or 300.0)
    except Exception:
        continue
    if not (fl & FL_ONGROUND):
        continue
    iv = inverse_ground(v, vn, ms)
    wd = iv['wishdir']
    if wd is None:
        continue
    wd = np.asarray(wd, dtype=float).ravel()[:2]
    base = np.array([v[0], v[1], 0.0])
    target = np.array([vn[0], vn[1]])

    def err(ws):
        out = ground_step(base, wd, ws, 1e9)
        w2 = out[0] if isinstance(out, tuple) else out
        return float(np.linalg.norm(np.array([w2[0], w2[1]]) - target))

    lo, hi = 1.0, 900.0
    if err(lo) < err(hi):          # monotone increasing in ws?
        continue
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if err(mid) > 0.01:
            hi = mid
        else:
            lo = mid
    need.append(lo)
    msp.append(ms)
A = np.array(need)
M = np.array(msp)
print('ground moving ticks solved: %d' % len(A))
if len(A) == 0:
    sys.exit()
print('required wishspeed: med %.2f  p10 %.2f  p90 %.2f  min %.2f max %.2f'
      % (np.median(A), np.percentile(A, 10), np.percentile(A, 90), A.min(), A.max()))
print('mspd from CSV     : med %.2f  uniq %s' % (np.median(M), np.unique(M)[:10].tolist()))
print('frac required < mspd: %.3f    required > mspd: %.3f'
      % ((A < M).mean(), (A > M).mean()))
h, edges = np.histogram(A, bins=[0, 100, 200, 250, 300, 350, 400, 450, 500, 900])
print('histogram:', {int(edges[i]): int(h[i]) for i in range(len(h))})