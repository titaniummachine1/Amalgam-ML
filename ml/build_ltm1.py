"""LTM-v1 (observable-only, per-engagement): fixed-size accumulator stats, no
predictions needed, deterministic from ticks DB. Engagement = current consecutive
same-class run containing T (gaps/teleports/class-change bound it; cap 600 ticks).
Per sample (8 feats): strafe_bias, reversal_rate, speed_mean/var, accel_var,
airtime_frac, run_len_norm, eye_stable_frac. Output aligned 1:1 with trainA/evalA.
DB: dataset_all.db. Run AFTER finetune (CPU busy)."""
import os
import numpy as np
import sqlite3
from collections import defaultdict

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DS = D
DT = 1 / 66.0
CAP = 600
con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))


def build(tag):
    k = np.load(os.path.join(D, tag + 'Ak.npz'))
    K = k['K']
    N = len(K)
    L = np.zeros((N, 8), dtype=np.float32)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(K[i][0]), int(K[i][1]))].append(i)
    for (did, ent), idxs in groups.items():
        rows = con.execute(
            'SELECT tick,ox,oy,grounded,eyey,cls FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
            (did, ent)).fetchall()
        if not rows:
            continue
        tk = np.array([r[0] for r in rows])
        ox = np.array([r[1] for r in rows])
        oy = np.array([r[2] for r in rows])
        gr = np.array([r[3] for r in rows], dtype=np.float64)
        ey = np.array([r[4] for r in rows])
        cl = np.array([r[5] for r in rows])
        pos = {t: j for j, t in enumerate(tk)}
        gap = np.diff(tk, prepend=tk[0] - 1)
        step = np.hypot(np.diff(ox, prepend=ox[0]), np.diff(oy, prepend=oy[0]))
        ok = (gap == 1) & (step <= 300.0) & (np.ediff1d(cl, to_begin=0) == 0)
        ok[0] = False
        vx = np.diff(ox, prepend=ox[0]) / np.maximum(gap, 1) / DT
        vy = np.diff(oy, prepend=oy[0]) / np.maximum(gap, 1) / DT
        for i in idxs:
            t = int(K[i][2])
            jT = pos.get(t)
            if jT is None:
                continue
            # walk back to engagement start
            js = jT
            while js - 1 >= 0 and ok[js] and (jT - (js - 1)) <= CAP:
                js -= 1
            seg = np.arange(js, jT + 1)
            if len(seg) < 8:
                continue
            vv = np.stack([vx[seg], vy[seg]], axis=1)
            sp = np.linalg.norm(vv, axis=1)
            hd = np.arctan2(vv[:, 1], vv[:, 0])
            dh = (np.diff(hd) + np.pi) % (2 * np.pi) - np.pi
            # strafe bias: sign consistency of turns (persistent L/R lean)
            bias = float(np.sign(dh).mean()) if len(dh) else 0.0
            # reversal rate: sign flips per second
            sgn = np.sign(dh)
            rev = float(((sgn[1:] * sgn[:-1]) < 0).sum() / max(len(dh) * DT, 1e-6))
            L[i, 0] = np.clip(bias, -1, 1)
            L[i, 1] = min(rev, 10.0) / 10.0
            L[i, 2] = min(sp.mean() / 400.0, 2.0) / 2.0
            L[i, 3] = min(sp.var() / 40000.0, 1.0)
            acc = np.diff(vv, axis=0) / DT
            L[i, 4] = min(np.var(acc) / 4e6, 1.0)
            L[i, 5] = float(1.0 - gr[seg].mean())
            L[i, 6] = min(len(seg) / CAP, 1.0)
            L[i, 7] = float((np.nanstd(ey[seg]) < 1.0))
    np.savez_compressed(os.path.join(D, tag + '_ltm1.npz'), L=L)
    print(tag, L.shape, 'frac nonzero: %.3f' % ((np.abs(L).sum(axis=1) > 0).mean()), flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
