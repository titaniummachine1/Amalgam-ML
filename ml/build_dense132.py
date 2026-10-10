"""Dense 132-tick target cache for soldier (every tick 1..132 gets a target).
Mirrors build_dense66.py exactly (same R, same validity), extended to 2s so the
correction-transformer + far-horizon training have per-tick supervision.
Validity per tick: consecutive ticks, same class (soldier), no teleport step.
Output: trainB_dense132.npz / evalB_dense132.npz with Hd (N,132,2), Vd (N,132 bool).
Eval keys are FROZEN (same evalBk); only the target extension is new."""
import os
import sqlite3
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DS = D
DT = 1 / 66.0
TELEPORT = 300.0
HOR132 = 132
EYE_FRAME = os.environ.get('EYE_FRAME', '0') == '1'
con = sqlite3.connect(os.path.join(DS, os.environ.get('ML_DB', 'dataset_all.db')))


def build(tag):
    SUF = os.environ.get('DS_SUFFIX', 'B')
    k = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    K = k[k[:, 3] == 3]
    N = len(K)
    assert N > 0, 'no soldier keys'
    Hd = np.full((N, HOR132, 2), np.nan, dtype=np.float32)
    Vd = np.zeros((N, HOR132), dtype=bool)
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
        consec = np.zeros(len(tk), dtype=bool)
        consec[1:] = (np.diff(tk) == 1) & (cl[1:] == 3)
        step = np.hypot(np.diff(ox, prepend=ox[0]), np.diff(oy, prepend=oy[0]))
        step_ok = np.ones(len(tk), dtype=bool)
        step_ok[1:] = step[1:] <= TELEPORT
        step_ok[0] = False
        valid_step = consec & step_ok
        bad = (~valid_step).astype(np.int64)
        cs = np.concatenate([[0], np.cumsum(bad)])
        for i in idxs:
            t = int(K[i][2])
            jT = pos.get(t)
            if jT is None or jT < 66:
                continue
            vlt = np.array([ox[jT] - ox[jT - 1], oy[jT] - oy[jT - 1]]) / DT
            yaw = ey[jT]
            if not np.isfinite(yaw):
                yaw = 0.0
            if EYE_FRAME:
                if not np.isfinite(ey[jT]):
                    yaw = (np.rad2deg(np.arctan2(vlt[1], vlt[0]))
                           if np.linalg.norm(vlt) > 1.0 else 0.0)
            elif np.linalg.norm(vlt) > 50:
                yaw = np.rad2deg(np.arctan2(vlt[1], vlt[0]))
            a = np.deg2rad(yaw)
            c, sn = np.cos(-a), np.sin(-a)
            R = np.array([[c, -sn], [sn, c]])
            o0x, o0y = ox[jT], oy[jT]
            for h in range(1, HOR132 + 1):
                jj = jT + h
                if jj >= len(tk) or tk[jj] != t + h:
                    continue
                if cs[jj + 1] - cs[jT + 1] > 0:
                    continue
                d = R @ np.array([ox[jj] - o0x, oy[jj] - o0y])
                Hd[i, h - 1, 0] = d[0]
                Hd[i, h - 1, 1] = d[1]
                Vd[i, h - 1] = True
        done += 1
        if done % 10 == 0:
            print('%s groups %d/%d' % (tag, done, len(groups)), flush=True)
    np.savez_compressed(os.path.join(D, tag + SUF + '_dense132.npz'), Hd=Hd, Vd=Vd)
    cov = Vd.mean(axis=0)
    print(tag, Hd.shape, 'coverage t=1/7/33/66/132:',
          '%.2f/%.2f/%.2f/%.2f/%.2f' % (cov[0], cov[6], cov[32], cov[65], cov[131]),
          flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
