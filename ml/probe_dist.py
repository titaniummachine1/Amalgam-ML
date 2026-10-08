"""Distance-to-nearest-damage histogram for high-|A| labels.
If clustered near events -> alignment/window issue. If uniform -> other source."""
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import didreg
import sqlite3
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DT = 1 / 66.0
con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))
import didreg as _dr
did2base = _dr.reverse(_dr.load())  # did -> basename, stable forever
files = None  # registry-owned

# damage tick sets per (base, ent) — same join as measure_drops (health correlation)
print('indexing damage...', flush=True)
import csv
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
                        dmg[(base, ent)].add(t)
                        break
                else:
                    continue
                break

# sample series, compute |A| with nearest-damage distance
import random
random.seed(2)
pairs = [(d, e) for (d, e) in con.execute('SELECT DISTINCT demo,ent FROM ticks').fetchall()]
pairs = random.sample(pairs, min(60, len(pairs)))
big_near = []
big_far_d = []
for did, ent in pairs:
    base = did2base.get(did)
    rows = con.execute(
        'SELECT tick,ox,oy,grounded,cls FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
        (did, ent)).fetchall()
    if len(rows) < 500:
        continue
    tk = np.array([r[0] for r in rows])
    ox = np.array([r[1] for r in rows])
    oy = np.array([r[2] for r in rows])
    gr = np.array([r[3] for r in rows])
    gap = np.diff(tk, prepend=tk[0] - 1)
    step = np.hypot(np.diff(ox, prepend=ox[0]), np.diff(oy, prepend=oy[0]))
    ok = (gap == 1) & (step <= 300.0)
    ok[0] = False
    vx = np.diff(ox, prepend=ox[0]) / np.maximum(gap, 1) / DT
    vy = np.diff(oy, prepend=oy[0]) / np.maximum(gap, 1) / DT
    v0 = np.stack([vx[:-1], vy[:-1]], axis=1)
    v1 = np.stack([vx[1:], vy[1:]], axis=1)
    okp = ok[:-1] & ok[1:]
    A = (v1 - v0) / DT
    sp0 = np.linalg.norm(v0, axis=1)
    gm = gr[:-1] > 0.5
    A[gm] += (v0[gm] / np.maximum(sp0[gm], 1e-6)[:, None]) * np.maximum(sp0[gm], 100.0)[:, None] * 4.0
    mag = np.linalg.norm(A, axis=1)
    ev = np.array(sorted(dmg.get((base, ent), set())))
    tt = tk[:-1]
    himag = np.where(okp & (mag > 3360))[0]
    for j in himag[::max(1, len(himag) // 400)]:
        if len(ev) == 0:
            big_far_d.append(9999)
        else:
            i = np.searchsorted(ev, tt[j])
            cand = []
            if i < len(ev):
                cand.append(abs(int(ev[i] - tt[j])))
            if i > 0:
                cand.append(abs(int(tt[j] - ev[i - 1])))
            big_far_d.append(min(cand) if cand else 9999)
big_far_d = np.array(big_far_d)
print(f'high-|A| samples: n={len(big_far_d)}', flush=True)
for b in [5, 10, 33, 100]:
    print(f'  within {b} ticks of damage: {(big_far_d <= b).mean():.3f}', flush=True)
print('  never near damage (>1000):', (big_far_d > 1000).mean(), flush=True)
