"""Auxiliary labels for Maia-style extra supervision (group: aux).
Per sample (keyed demo/ent/T from trainAk/evalAk):
  aux_gr20 : grounded flag at T+20 (0/1)
  aux_sp20 : speed at T+20, normalized by class MAXV
  aux_turn : max |heading change| over next 20 ticks, /pi
Missing (gap/teleport/class-change within window) -> weight 0.
Output: trainA_aux.npz / evalA_aux.npz with A3 (N,3) + M (N,3) mask.
DB: dataset_all.db (grounded/positions), keys give class for MAXV.
"""
import os
import numpy as np
import sqlite3
from collections import defaultdict

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DS = D
DT = 1 / 66.0
MAXV_CLS = {1: 400.0, 2: 300.0, 3: 240.0, 4: 280.0, 5: 320.0,
            6: 230.0, 7: 300.0, 8: 300.0, 9: 300.0}
H_AUX = 20
con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))


def build(tag):
    k = np.load(os.path.join(D, tag + 'Ak.npz'))
    K = k['K']
    N = len(K)
    A3 = np.zeros((N, 3), dtype=np.float32)
    M = np.zeros((N, 3), dtype=np.float32)
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
        gr = np.array([r[3] for r in rows])
        cl = np.array([r[4] for r in rows])
        pos = {t: j for j, t in enumerate(tk)}
        for i in idxs:
            t = int(K[i][2])
            cls = int(K[i][3])
            mv = MAXV_CLS.get(cls, 300.0)
            jT = pos.get(t)
            if jT is None:
                continue
            jF = pos.get(t + H_AUX)
            if jF is None or tk[jF] != t + H_AUX:
                continue
            # validity: consecutive ticks, same class, no teleport across (T, T+20]
            seg = np.arange(jT + 1, jF + 1)
            if len(seg) == 0:
                continue
            if not np.all(tk[seg] == np.arange(t + 1, t + H_AUX + 1)):
                continue
            if np.any(cl[seg] != cl[jT]):
                continue
            st = np.hypot(ox[seg] - ox[seg - 1], oy[seg] - oy[seg - 1])
            if (st > 300.0).any():
                continue
            A3[i, 0] = float(gr[jF] > 0.5)
            vx = (ox[jF] - ox[jF - 1]) / DT
            vy = (oy[jF] - oy[jF - 1]) / DT
            A3[i, 1] = float(np.hypot(vx, vy) / mv)
            hd = np.arctan2(oy[seg] - oy[seg - 1], ox[seg] - ox[seg - 1])
            dh = (np.diff(hd) + np.pi) % (2 * np.pi) - np.pi
            A3[i, 2] = float(np.abs(dh).max(initial=0.0) / np.pi) if len(dh) else 0.0
            M[i, :] = 1.0
    np.savez_compressed(os.path.join(D, tag + '_aux.npz'), A3=A3, M=M)
    print(tag, A3.shape, 'mask coverage:', M.mean(axis=0).round(3), flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
