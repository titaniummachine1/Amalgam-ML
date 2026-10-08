"""Precompute loadout id columns (wld0-2, uint8) for E arrays -- memory-capped.

trainE.npz on disk is 18-dim [14 base + wslot(3) + wcls(1)] and stays as-is.
This script ONLY builds the 3 missing loadout-id columns from dataset_eye.db
and saves trainE_wld.npz / evalE_wld.npz (uint8 class ids 0-9, same row order
as trainE/EvalE keys). The trainer concats per batch: [X18 | wld/16] = 21-dim.
Nothing materializes more than one chunk: X untouched (mmap not even opened
here), wld written straight to a disk-backed memmap.

CONTRACT (must match build_ds_all.py WCLASS exactly):
  rocket:1 DH:2 AirStrike:3 CowMangler:4 soldierShotgun:5 BuffItem:6
  Shovel:7 Katana:8 Sword:9 ; wld[k] = id of w{0,1,2} column class (0 unknown)
Window ticks per key are consecutive ints ending at label-1 (builder
enforces consecutive runs), so ticks = T-66..T-1.
"""
import numpy as np
import os
import sqlite3
import sys
from collections import defaultdict

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DB = os.path.join(D, 'dataset_eye.db')
W = 66
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
TMP = r'C:\Users\TERMIN~1\AppData\Local\Temp\opencode'


def process(tag):
    K = np.load(os.path.join(D, tag + 'Ek.npz'))['K']
    N = K.shape[0]
    print(f'{tag}: N={N}', flush=True)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(K[i][0]), int(K[i][1]))].append(i)
    print(f'{tag}: groups={len(groups)}', flush=True)

    outp = os.path.join(TMP, tag + 'Ewld mmap.tmp')
    if os.path.exists(outp):
        os.remove(outp)
    out = np.memmap(outp, dtype=np.uint8, mode='w+', shape=(N, W, 3))

    con = sqlite3.connect(DB)
    done = 0
    for (did, ent), idxs in groups.items():
        idxs = np.array(idxs)
        T = K[idxs, 2].astype(int)
        lo, hi = int(T.min()) - W, int(T.max())
        m = {}
        for t, a, b, c in con.execute(
                'SELECT tick,w0,w1,w2 FROM ticks WHERE demo=? AND ent=? AND tick>=? AND tick<=? ORDER BY tick',
                (did, ent, lo, hi)).fetchall():
            ia = WCLASS.get(a or '')
            ib = WCLASS.get(b or '')
            ic = WCLASS.get(c or '')
            m[t] = (ia[1] if ia else 0, ib[1] if ib else 0, ic[1] if ic else 0)
        for ii, t in zip(idxs, T):
            w = np.zeros((W, 3), dtype=np.uint8)
            for h in range(W):
                v = m.get(t - W + h)
                if v:
                    w[h] = v
            out[ii] = w
        done += len(idxs)
        if done % 400_000 < len(idxs):
            print(f'{tag}: {done}/{N}', flush=True)
    con.close()
    out.flush()
    dst = os.path.join(D, tag + 'E_wld.npz')
    np.savez_compressed(dst, WLD=np.array(out))
    del out
    os.remove(outp)
    print(f'{tag}: wrote {dst}', flush=True)


if __name__ == '__main__':
    for tag in sys.argv[1:] or ['train', 'eval']:
        process(tag)
    print('WLD DONE', flush=True)
