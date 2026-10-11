import os

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
H = np.load(os.path.join(D, 'trainB_dense132.npz'), mmap_mode='r')['Hd']
V = np.load(os.path.join(D, 'trainB_dense132.npz'), mmap_mode='r')['Vd']
X = np.memmap(os.path.join(D, 'trainB_X20sol.raw'), dtype=np.float32, mode='r',
              shape=(1340432, 66, 20))
k = np.load(os.path.join(D, 'trainBk.npz'))['K']
sol = np.where(k[:, 3] == 3)[0]
print('keys:', [(int(k[sol[i]][0]), int(k[sol[i]][1]), int(k[sol[i]][2])) for i in [320178, 320180, 320184]])
h = np.array(H[320178:320185])
print('Hd finite:', bool(np.isfinite(h).all()), 'nan frac:', float(np.isnan(h).mean()))
print('Hd[2,70:80,0]:', h[2, 70:80, 0])
print('V[320180] sum:', int(np.array(V[320180]).sum()))
print('v0 [320180]:', (np.array(X[320180, -1, :2]) * 240).tolist())
print('Hd[0,:,0] first/last:', float(h[0, 0, 0]), float(h[0, -1, 0]))
