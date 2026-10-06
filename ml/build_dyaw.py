"""Per-tick eye-yaw-delta labels for the input-space policy (dyaw has GT, fm/sm do not).

For every soldier sample (aligned 1:1 with the dense66 / threatwin caches, i.e. the
cls==3 rows of trainBk/evalBk in file order) emit:
  Dy (N,66) float32  per-tick eye-yaw delta in DEGREES, wrapped to (-180,180],
                     NaN where the tick is missing / non-finite / non-consecutive
  Fr (N,)   bool     eye FRESH over the future window [t, t+66]: nanstd > 0.5.
                     Only these rows may be used for the dyaw aux loss (demo eye
                     goes stale -> deltas are frozen garbage, not "player held still")
  FS (N,)   uint8    ego-frame source at T: 1 = eye yaw, 0 = velocity-heading
                     fallback (mirrors build_ds_all / build_dense66 rule)

Units note: DB eyey is degrees. Model head is tanh -> deg = tanh*DYAW_SCALE.
Output: trainB_dyaw.npz / evalB_dyaw.npz
"""
import os
import sqlite3
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DB = os.path.join(D, os.environ.get('ML_DB', 'dataset_all.db'))
SUF = os.environ.get('OUT_SUFFIX', 'B')
W = 66
EYE_STD_MIN = 0.5


def build(tag):
    k = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    N = len(Ks)
    Dy = np.full((N, W), np.nan, dtype=np.float32)
    Dp = np.full((N, W), np.nan, dtype=np.float32)
    Fr = np.zeros(N, dtype=bool)
    FS = np.zeros(N, dtype=np.uint8)
    con = sqlite3.connect(DB)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(Ks[i][0]), int(Ks[i][1]))].append(i)
    done = 0
    h = np.arange(1, W + 1)
    for (did, ent), idxs in groups.items():
        rows = con.execute(
            'SELECT tick,ox,oy,eyey,eyep FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
            (did, ent)).fetchall()
        if not rows:
            continue
        tk = np.array([r[0] for r in rows])
        ox = np.array([r[1] for r in rows])
        oy = np.array([r[2] for r in rows])
        ey = np.array([r[3] for r in rows], dtype=np.float64)
        ep = np.array([r[4] if r[4] is not None else np.nan for r in rows],
                      dtype=np.float64)
        pos = {t: j for j, t in enumerate(tk)}
        for i in idxs:
            t = int(Ks[i][2])
            jT = pos.get(t)
            if jT is None or jT < 1:
                continue
            # --- ego-frame source at T (identical rule to build_ds_all:
            # velocity heading when moving, else eye; no std threshold)
            vlt = np.array([ox[jT] - ox[jT - 1], oy[jT] - oy[jT - 1]]) * 66.0
            sp = float(np.hypot(*vlt))
            if sp > 50.0:
                FS[i] = 0
            else:
                FS[i] = 1
            # --- future window
            jj = jT + h
            if jj[-1] >= len(tk):
                continue
            if not np.array_equal(tk[jj], t + h):
                continue  # DB gap inside the future window: whole row unusable
            fut = ey[jj]
            if not np.isfinite(fut).all() or not np.isfinite(ey[jT]):
                continue
            d = np.diff(np.concatenate([[ey[jT]], fut]))
            d = (d + 180.0) % 360.0 - 180.0
            Dy[i] = d
            fp = ep[jj]
            if np.isfinite(fp).all() and np.isfinite(ep[jT]):
                Dp[i] = np.diff(np.concatenate([[ep[jT]], fp]))
            Fr[i] = bool(np.std(np.concatenate([[ey[jT]], fut])) > EYE_STD_MIN)
        done += 1
        if done % 200 == 0:
            print(f'{tag} groups {done}/{len(groups)}', flush=True)
    np.savez_compressed(os.path.join(D, tag + SUF + '_dyaw.npz'), Dy=Dy, Dp=Dp, Fr=Fr, FS=FS)
    print(tag, 'Dy', Dy.shape,
          'nan rows %.4f' % np.isnan(Dy[:, 0]).mean(),
          'eye-fresh frac %.3f' % Fr.mean(),
          'frame=eye frac %.3f' % (FS == 1).mean(), flush=True)
    m = np.isfinite(Dy)
    a = np.abs(Dy[m])
    if len(a):
        print(tag, '|dyaw| deg p50 %.3f p90 %.3f p99 %.3f p99.9 %.3f max %.1f' % tuple(
            np.percentile(a, [50, 90, 99, 99.9, 100])), flush=True)
        print(tag, 'mean |dyaw| deg/tick %.4f  -> 66-tick total %.1f deg' % (a.mean(), a.mean() * 66),
              flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
