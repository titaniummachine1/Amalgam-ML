"""Angular modality of pure-input accel labels (trainB, ego frame, x=yaw-forward).
Are human input accels concentrated near 8 compass dirs (0,45,...)? Reports
angular histogram + mass within +-11.25deg of each mode vs uniform (25%)."""
import os
import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
tr = np.load(os.path.join(D, 'trainB.npz'), mmap_mode='r')
Y = np.array(tr['Y'])
W = np.array(tr['W'])
m = np.linalg.norm(Y, axis=1) > 0.5  # moving only; idle has no direction
Y = Y[m]
W = W[m]
ang = np.arctan2(Y[:, 1], Y[:, 0])  # ego frame: 0 = yaw-forward
bins = np.linspace(-np.pi, np.pi, 17)
hist, _ = np.histogram(ang, bins=bins, weights=W)
hist = hist / hist.sum() * 100
names = ['-180', '-157', '-135', '-112', '-90', '-67', '-45', '-22', '0',
         '22', '45', '67', '90', '112', '135', '157']
print('angular %% (weighted, 22.5deg bins, 0=yaw-fwd):')
for n, h in zip(names, hist):
    print(f'  {n:>5}: {h:5.1f}%')
modes = np.array([0, 45, 90, 135, 180, -135, -90, -45]) * np.pi / 180
d = np.abs((ang[:, None] - modes[None, :] + np.pi) % (2 * np.pi) - np.pi).min(axis=1)
print('mass within +-11.25deg of a mode: %.1f%% (uniform: 25%%)' % ((d < np.pi / 16).mean() * 100))
print('n =', len(Y))
