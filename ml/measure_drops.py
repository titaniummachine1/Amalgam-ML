"""Are impulse-split drops legit knockback or over-tight caps?
Recompute raw |A| for candidate labels (trainA windows), split by near-damage.
Needs: trainA keys + DB. Reports |A| percentiles near vs far from damage."""
import glob
import os
import sqlite3
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DT = 1 / 66.0
MAXV_CLS = {1: 400.0, 2: 300.0, 3: 240.0, 4: 280.0, 5: 320.0,
            6: 230.0, 7: 300.0, 8: 300.0, 9: 300.0}
con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))

import csv
files = sorted(f for f in glob.glob(os.path.join(D, '*.csv'))
               if not f.endswith('.proj.csv') and not f.endswith('.hurt.csv'))
did2base = {i: os.path.basename(f) for i, f in enumerate(files)}

# damage tick sets per (base, ent) via health-drop join (reuse logic, compact)
print('indexing damage...', flush=True)
dmg = defaultdict(set)
for base in set(did2base.values()):
    health = defaultdict(dict)
    with open(os.path.join(D, base)) as fh:
        for row in csv.DictReader(fh):
            if row['life'] != '0':
                continue
            try:
                health[int(row['ent'])][int(row['tick'])] = int(row['health'])
            except ValueError:
                continue
    hf = os.path.join(D, base + '.hurt.csv')
    if not os.path.exists(hf):
        continue
    with open(hf) as fh:
        for row in csv.DictReader(fh):
            if row['kind'] != 'hurt':
                continue
            t, dmgv, posthp = int(row['tick']), int(row['c']), int(row['e'])
            for ent, hs in health.items():
                for dt in range(0, 4):
                    h = hs.get(t + dt)
                    if h is None:
                        continue
                    if h == posthp or (hs.get(t + dt - 1) is not None
                                       and hs[t + dt - 1] - h >= max(dmgv - 15, 1)):
                        for ddt in range(-5, 6):
                            dmg[(base, ent)].add(t + ddt)
                        break
                else:
                    continue
                break

# candidate labels: stride through trainA keys is overkill; sample series directly.
# For each (demo,ent) series: consecutive ok ticks, compute |A_raw| + near-damage flag.
k = np.load(os.path.join(DS, 'trainAk.npz'))['K']
K = k
print('sampling series...', flush=True)
near_vals = []
far_vals = []
import random
random.seed(1)
pairs = list({(int(K[i][0]), int(K[i][1])) for i in range(len(K))})
pairs = random.sample(pairs, min(120, len(pairs)))
for did, ent in pairs:
    base = did2base.get(did)
    rows = con.execute(
        'SELECT tick,ox,oy,grounded,cls FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
        (did, ent)).fetchall()
    if len(rows) < 200:
        continue
    tk = np.array([r[0] for r in rows])
    ox = np.array([r[1] for r in rows])
    oy = np.array([r[2] for r in rows])
    gr = np.array([r[3] for r in rows])
    cl = np.array([r[4] for r in rows])
    gap = np.diff(tk, prepend=tk[0] - 1)
    step = np.hypot(np.diff(ox, prepend=ox[0]), np.diff(oy, prepend=oy[0]))
    ok = (gap == 1) & (step <= 300.0) & (np.ediff1d(cl, to_begin=0) == 0)
    ok[0] = False
    vx = np.diff(ox, prepend=ox[0]) / np.maximum(gap, 1) / DT
    vy = np.diff(oy, prepend=oy[0]) / np.maximum(gap, 1) / DT
    v = np.stack([vx, vy], axis=1)
    v0 = v[:-1]
    v1 = v[1:]
    okp = ok[:-1] & ok[1:]
    A = (v1 - v0) / DT
    sp0 = np.linalg.norm(v0, axis=1)
    gm = gr[:-1] > 0.5
    f = np.maximum(sp0, 100.0) * 4.0
    A[gm] += (v0[gm] / np.maximum(sp0[gm], 1e-6)[:, None]) * f[gm][:, None]
    mag = np.linalg.norm(A, axis=1)
    cls0 = cl[:-1]
    dset = dmg.get((base, ent), set())
    ticks = tk[:-1]
    isnear = np.array([t in dset for t in ticks])
    for clsid in (3,):
        m = okp & (cls0 == clsid)
        near_vals += mag[m & isnear].tolist()
        far_vals += mag[m & ~isnear].tolist()
near_vals = np.array(near_vals)
far_vals = np.array(far_vals)
print(f'near-damage labels: n={len(near_vals)}', flush=True)
print(f'far labels: n={len(far_vals)}', flush=True)
for name, v in [('near', near_vals), ('far', far_vals)]:
    if len(v) == 0:
        continue
    print(f'{name}: p50={np.percentile(v,50):.0f} p90={np.percentile(v,90):.0f} '
          f'p99={np.percentile(v,99):.0f} p999={np.percentile(v,99.9):.0f} '
          f'frac>3360={(v > 3360).mean():.3f} frac>1500(air-equiv)={(v > 1500).mean():.3f}', flush=True)
