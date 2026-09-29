"""Dense 66-tick target cache for soldier windows (trainA/evalA, class 3).
For every sample: actual ego-frame displacement at ticks 1..66 + validity mask.
Validity per tick: consecutive ticks, same class, no teleport step.
R recomputed EXACTLY as build_ds_all.py; validated against stored H.
Output: trainA_dense66.npz / evalA_dense66.npz with Hd (N,66,2), Vd (N,66 bool)."""
import os
import numpy as np
import sqlite3
from collections import defaultdict

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DS = D
DT = 1 / 66.0
TELEPORT = 300.0
con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))


def build(tag):
    k = np.load(os.path.join(D, tag + 'Ak.npz'))
    K = k['K']
    sel = K[:, 3] == 3
    K = K[sel]
    N = len(K)
    Hd = np.full((N, 66, 2), np.nan, dtype=np.float32)
    Vd = np.zeros((N, 66), dtype=bool)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(K[i][0]), int(K[i][1]))].append(i)
    done = 0
    for (did, ent), idxs in groups.items():
        rows = con.execute(
            'SELECT tick,ox,oy,eyey,cls FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
            (did, ent)).fetchall()
        if not rows:
            continue
        tk = np.array([r[0] for r in rows])
        ox = np.array([r[1] for r in rows])
        oy = np.array([r[2] for r in rows])
        ey = np.array([r[3] for r in rows])
        cl = np.array([r[4] for r in rows])
        pos = {t: j for j, t in enumerate(tk)}
        # per-step validity across whole series (vectorized once per group)
        consec = np.zeros(len(tk), dtype=bool)
        consec[1:] = (np.diff(tk) == 1) & (cl[1:] == 3)
        step = np.hypot(np.diff(ox, prepend=ox[0]), np.diff(oy, prepend=oy[0]))
        step_ok = np.ones(len(tk), dtype=bool)
        step_ok[1:] = step[1:] <= TELEPORT
        step_ok[0] = False
        valid_step = consec & step_ok
        # cumsum of invalid for O(1) range queries
        bad = (~valid_step).astype(np.int64)
        cs = np.concatenate([[0], np.cumsum(bad)])
        for i in idxs:
            t = int(K[i][2])
            jT = pos.get(t)
            if jT is None or jT < 66:
                continue
            # R identical to build_ds_all.py
            vlt = np.array([ox[jT] - ox[jT - 1], oy[jT] - oy[jT - 1]]) / DT
            wy = ey[jT - 65:jT + 1]
            yaw = ey[jT]
            if not np.isfinite(yaw):
                yaw = 0.0
            if np.nanstd(wy) < 0.5 and np.linalg.norm(vlt) > 50:
                yaw = np.rad2deg(np.arctan2(vlt[1], vlt[0]))
            a = np.deg2rad(yaw)
            c, sn = np.cos(-a), np.sin(-a)
            R = np.array([[c, -sn], [sn, c]])
            o0x, o0y = ox[jT], oy[jT]
            # all 66 future ticks: positional index + tick-number guard (catches DB gaps)
            for h in range(1, 67):
                jj = jT + h
                # verify tick number matches (guards DB gaps): tk[jj] must equal t+h
                if jj >= len(tk) or tk[jj] != t + h:
                    continue
                if cs[jj + 1] - cs[jT + 1] > 0:
                    continue  # some step in (T, T+h] invalid
                d = R @ np.array([ox[jj] - o0x, oy[jj] - o0y])
                Hd[i, h - 1, 0] = d[0]
                Hd[i, h - 1, 1] = d[1]
                Vd[i, h - 1] = True
        done += 1
        if done % 10 == 0:
            print(f'{tag} groups {done}/{len(groups)}', flush=True)
    np.savez_compressed(os.path.join(D, tag + 'A_dense66.npz'), Hd=Hd, Vd=Vd)
    cov = Vd.mean(axis=0)
    print(tag, Hd.shape, 'coverage t=1/7/33/66:',
          f'{cov[0]:.2f}/{cov[6]:.2f}/{cov[32]:.2f}/{cov[65]:.2f}', flush=True)
    # validate R: horizons [7,13,20,33,66] (idx 6,12,19,32,65) vs stored H
    d = np.load(os.path.join(D, tag + 'A.npz'), mmap_mode='r')
    H = np.array(d['H'])
    kfull = np.load(os.path.join(D, tag + 'Ak.npz'))['K']
    H = H[kfull[:, 3] == 3]
    err = []
    for j, hi in enumerate([6, 12, 19, 32, 65]):
        a, b = Hd[:, hi], H[:, j]
        m = Vd[:, hi] & ~np.isnan(b[:, 0])
        if m.sum():
            err.append(float(np.abs(a[m] - b[m]).max()))
    print(tag, 'max |recomputed - stored| on shared horizons:', err, flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
