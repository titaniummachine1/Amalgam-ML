"""Weapon-7 cache for the combined threat+weapon model (B line, soldier rows).
Per row key (did,ent,T): 66 warmup ticks [T-66,T-1] of [wslot(3) + wcls(1) +
wld(3)] mapped EXACTLY like build_ds_all WEAPON_DIMS (same WCLASS table).
Source: dataset_eye.db (wcls/w0/w1/w2 cols, forward-filled by build_db_eye).
Rows from demos absent in eye DB (4th pull, eye never re-ingested) -> zeros
= unknown bucket (training convention). Output trainB_w7.npz / evalB_w7.npz
with W7 (N,66,7) float32, soldier-ordered 1:1 with repack/dense caches.
Usage: python ml/build_w7.py [train|eval|both]"""
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
con = sqlite3.connect(os.path.join(D, 'dataset_eye.db'))
WCLASS = {
    'CTFRocketLauncher': (0, 1),
    'CTFRocketLauncher_DirectHit': (0, 2),
    'CTFRocketLauncher_AirStrike': (0, 3),
    'CTFParticleCannon': (0, 4),
    'CTFShotgun_Soldier': (1, 5),
    'CTFBuffItem': (1, 6),
    'CTFShovel': (2, 7),
    'CTFKatana': (2, 8),
    'CTFSword': (2, 9),
}
WCLS_SCALE = 16.0
W = 66


def build(tag):
    k = np.load(os.path.join(D, tag + 'Bk.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    N = len(Ks)
    W7 = np.zeros((N, W, 7), dtype=np.float32)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(Ks[i][0]), int(Ks[i][1]))].append(i)
    done = 0
    hit = 0
    for (did, ent), idxs in groups.items():
        rows = con.execute(
            'SELECT tick,wcls,w0,w1,w2 FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
            (did, ent)).fetchall()
        if rows:
            hit += 1
            L = len(rows)
            S7 = np.zeros((L, 7), dtype=np.float32)
            # per-tick map (series-level loop; windows slice below vectorized)
            for j, r in enumerate(rows):
                se = WCLASS.get(r[1] or '')
                if se is not None and se[0] is not None:
                    S7[j, se[0]] = 1.0
                    S7[j, 3] = se[1] / WCLS_SCALE
                for kk in range(3):
                    se2 = WCLASS.get(r[2 + kk] or '')
                    if se2 is not None:
                        S7[j, 4 + kk] = se2[1] / WCLS_SCALE
            tk = np.array([r[0] for r in rows])
            pos = {t: j for j, t in enumerate(tk)}
            for i in idxs:
                T = int(Ks[i][2])
                a = pos.get(T - W, -1)
                if a >= 0 and a + W <= L and tk[a + W - 1] == T - 1:
                    W7[i] = S7[a:a + W]
        done += 1
        if done % 20 == 0:
            print('%s groups %d/%d eye-hit=%d' % (tag, done, len(groups), hit), flush=True)
    np.savez_compressed(os.path.join(D, tag + 'B_w7.npz'), W7=W7)
    print(tag, W7.shape, 'nonzero frac=%.4f groups-with-eye=%d/%d' % (
        (W7 != 0).mean(), hit, len(groups)), flush=True)


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'both'
    for t in (['train', 'eval'] if arg == 'both' else [arg]):
        build(t)
