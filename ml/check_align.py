import os
import sqlite3

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
con = sqlite3.connect(os.path.join(D, 'dataset_all.db'))
DT = 1 / 66.0
H = np.load(os.path.join(D, 'trainB_dense132.npz'), mmap_mode='r')['Hd']
V = np.load(os.path.join(D, 'trainB_dense132.npz'), mmap_mode='r')['Vd']
X = np.memmap(os.path.join(D, 'trainB_X20sol.raw'), dtype=np.float32, mode='r',
              shape=(1340432, 66, 20))
k = np.load(os.path.join(D, 'trainBk.npz'))['K']
sol = np.where(k[:, 3] == 3)[0]
for i in [320178, 320180, 320184, 146581]:
    did, ent, T = (int(k[sol[i]][0]), int(k[sol[i]][1]), int(k[sol[i]][2]))
    print('row', i, 'key', (did, ent, T))
    s = con.execute(
        'SELECT tick,ox,oy,eyey FROM ticks WHERE demo=? AND ent=? AND tick BETWEEN ? AND ? ORDER BY tick',
        (did, ent, T - 70, T + 5)).fetchall()
    tk = np.array([r[0] for r in s])
    jT = int(np.where(tk == T)[0][0]) if (tk == T).any() else -1
    print('  jT in db window:', jT, 'nticks:', len(tk))
    if jT >= 1:
        ox = np.array([r[1] for r in s])
        oy = np.array([r[2] for r in s])
        vlt = np.array([ox[jT] - ox[jT - 1], oy[jT] - oy[jT - 1]]) / DT
        print('  db v@T:', vlt.round(1).tolist(), 'yaw:', s[jT][3])
    h = np.array(H[i, :3])
    print('  Hd[:3]:', np.round(h, 1).tolist(), 'Vsum:', int(np.array(V[i]).sum()))
    print('  X v0:', (np.array(X[i, -1, :2]) * 240).round(1).tolist())
