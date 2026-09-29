"""Correction-history features (group: corr1).

Frames (all deterministic from ticks DB, replicable live):
  B = builder frame at sample tick (@T): eye yaw, or velocity-heading fallback
      (eye std over warmup < 0.5 and speed > 50) — identical rule to build_ds3.
  H = velocity-heading frame at T (fresh; used for corr inputs).
1-step pred A (in B frame, per-sec/MAXV norm) from champion v3ii, single batched pass.
Absolute predicted pos (world) vs absolute actual -> world error; both rotated to H.
Per sample, on-grid lags L={4,8,12,16,24,32}: err_H(2) + predacc_H(2). Missing -> 0.
Output C aligned 1:1 with train3/ev3 rows.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import sqlite3

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DT = 1 / 66.0
LAGS = [4, 8, 12, 16, 24, 32]


class Net(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.gru = nn.GRU(5, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


net = Net()
net.load_state_dict(torch.load(os.path.join(D, 'gru64_v3ii.pt'), map_location='cpu'))
net.eval()
con = sqlite3.connect(os.path.join(D, 'dataset.db'))


def track(did, ent, t0, t1):
    """Return dict tick->(ox,oy,ey) for range, plus ordered tick list."""
    rows = con.execute(
        'SELECT tick,ox,oy,eyey FROM ticks WHERE demo=? AND ent=? AND tick>=? AND tick<=? ORDER BY tick',
        (did, ent, t0, t1)).fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def builder_yaw(ey_warm, vlt):
    if np.nanstd(ey_warm) < 0.5 and np.linalg.norm(vlt) > 50:
        return np.rad2deg(np.arctan2(vlt[1], vlt[0]))
    y = ey_warm[-1]
    return 0.0 if not np.isfinite(y) else y


def build(tag):
    d = np.load(os.path.join(D, tag + '3.npz'), mmap_mode='r')
    k = np.load(os.path.join(D, tag + '3k.npz'))
    X, K = d['X'], k['K']
    N = len(X)
    A = np.zeros((N, 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, 1024):
            A[i:i + 1024] = net(torch.from_numpy(np.array(X[i:i + 1024]))).numpy()
    C = np.zeros((N, len(LAGS) * 4), dtype=np.float32)
    # group sample indices by (did,ent); load each series once
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(K[i][0]), int(K[i][1]))].append(i)
    for (did, ent), idxs in groups.items():
        rows = con.execute(
            'SELECT tick,ox,oy,eyey FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
            (did, ent)).fetchall()
        if not rows:
            continue
        tk = np.array([r[0] for r in rows])
        ox = np.array([r[1] for r in rows])
        oy = np.array([r[2] for r in rows])
        ey = np.array([r[3] for r in rows])
        pos = {t: j for j, t in enumerate(tk)}
        for i in idxs:
            t = int(K[i][2])
            jT = pos.get(t)
            if jT is None or jT < 1:
                continue
            vT = np.array([ox[jT] - ox[jT - 1], oy[jT] - oy[jT - 1]]) / DT
            spT = np.linalg.norm(vT)
            hyaw = np.rad2deg(np.arctan2(vT[1], vT[0])) if spT > 20 else 0.0
            an = np.deg2rad(hyaw)
            c, s = np.cos(-an), np.sin(-an)
            Rh = np.array([[c, -s], [s, c]])
            for li, L in enumerate(LAGS):
                jl = pos.get(t - L)
                if jl is None or jl < 67 or jl + 1 >= len(tk):
                    continue
                if tk[jl] - tk[jl - 66] != 66:
                    continue  # warmup must be consecutive
                wy = ey[jl - 65:jl + 1]
                if np.isnan(wy).any():
                    continue
                vlt = np.array([ox[jl] - ox[jl - 1], oy[jl] - oy[jl - 1]]) / DT
                byaw = builder_yaw(wy, vlt)
                ab = np.deg2rad(byaw)
                cb, sb = np.cos(ab), np.sin(ab)
                RbT = np.array([[cb, -sb], [sb, cb]])
                j = keymap.get((did, ent, t - L))
                if j is None:
                    continue
                aw = RbT @ (A[j] * 240.0)
                oxp = ox[jl] + vlt[0] * DT + 0.5 * aw[0] * DT * DT
                oyp = oy[jl] + vlt[1] * DT + 0.5 * aw[1] * DT * DT
                err = np.array([oxp - ox[jl + 1], oyp - oy[jl + 1]])
                C[i, li * 4:li * 4 + 2] = Rh @ err
                C[i, li * 4 + 2:li * 4 + 4] = Rh @ aw / 240.0
    np.savez_compressed(os.path.join(D, tag + '_corr1.npz'), C=C)
    print(tag, C.shape, 'frac with any corr: %.3f' % ((np.abs(C).sum(axis=1) > 0).mean()), flush=True)


# keymap built per tag inside build; need keys first: global pass
import sys
tag = sys.argv[1] if len(sys.argv) > 1 else 'train'
k = np.load(os.path.join(D, tag + '3k.npz'))
K = k['K']
keymap = {(int(K[i][0]), int(K[i][1]), int(K[i][2])): i for i in range(len(K))}
build(tag)
