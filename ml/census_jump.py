"""Census takeoff vz to set the jump threshold (jump impulse vs walked-off-ledge).
Samples soldier series from dataset_all.db: takeoff = gr[t-1]==1 & gr[t]==0.
Prints vz histogram at takeoff. Usage: python ml/census_jump.py"""
import os
import sqlite3

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
con = sqlite3.connect(os.path.join(D, 'dataset_all.db'))
DT = 1 / 66.0

dids = [r[0] for r in con.execute('SELECT DISTINCT demo FROM ticks').fetchall()][:40]
vz = []
n_fall = 0
for did in dids:
    ents = [r[0] for r in con.execute('SELECT DISTINCT ent FROM ticks WHERE demo=?', (did,))]
    for ent in ents:
        rows = con.execute(
            'SELECT tick,oz,grounded,cls FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
            (did, ent)).fetchall()
        if len(rows) < 200:
            continue
        tk = np.array([r[0] for r in rows])
        oz = np.array([r[1] for r in rows])
        gr = np.array([r[2] for r in rows])
        cl = np.array([r[3] for r in rows])
        if (cl == 3).sum() < 100:
            continue
        v = np.diff(oz, prepend=oz[0]) / DT
        take = np.where((gr[:-1] == 1) & (gr[1:] == 0))[0] + 1
        take = take[(tk[take] - tk[take - 1] == 1)]
        vz.extend(v[take].tolist())
vz = np.array(vz)
print('takeoffs:', len(vz))
edges = [-500, -50, 0, 50, 100, 150, 200, 250, 280, 320, 500, 2000]
h, _ = np.histogram(vz, bins=edges)
for lo, hi, c in zip(edges[:-1], edges[1:], h):
    print('%7.0f..%7.0f  %d' % (lo, hi, c))
