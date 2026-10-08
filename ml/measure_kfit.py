"""Fit K: observed |dv - friction| regressed on damage for DIRECTED hurts.

The forces rollout applies impulse = dir * dmg * K. This measures K from data
instead of guessing. Uses least squares with intercept (|dv| = K*dmg + c, where
c absorbs typical input accel); reports R^2 so we know whether to trust it.
If R^2 is noise, rollout runs k_imp=0 (feature-only) and the GRU carries it.
Usage: python ml/measure_kfit.py <tag> <suf: B|C>"""
import os
import sys

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DT = 1 / 66.0
tag = sys.argv[1] if len(sys.argv) > 1 else 'eval'
suf = sys.argv[2] if len(sys.argv) > 2 else 'B'

d = np.load(os.path.join(D, tag + suf + '.npz'), mmap_mode='r')
k = np.load(os.path.join(D, tag + suf + 'k.npz'))['K']
dn = np.load(os.path.join(D, tag + suf + '_dense66.npz'))
kb = np.load(os.path.join(D, tag + suf + '_knock.npz'))
sol = np.where(k[:, 3] == 3)[0]
X = np.array(d['X'])[sol]
Hd, Vd = np.array(dn['Hd']), np.array(dn['Vd'])
KB = np.array(kb['KB'])
assert KB.shape[0] == len(sol) == Hd.shape[0], (KB.shape, len(sol), Hd.shape)
gr = X[:, -1, 3] > 0.5  # grounded at T (approx for window)

# velocities per tick from dense displacements: v_h = (Hd[h]-Hd[h-1])/DT
pos0 = np.zeros_like(Hd[:, 0:1])
dpos = np.concatenate([Hd[:, 0:1] - pos0, Hd[:, 1:] - Hd[:, :-1]], axis=1)
vel = dpos / DT  # (N,66,2), NaN where invalid
# dv_h = v_h - friction(v_{h-1}); v_{-1} approx = v_0 - (v_1 - v_0)? use v0 for h=0
vprev = np.concatenate([vel[:, 0:1], vel[:, :-1]], axis=2) if False else None
v_prev = np.concatenate([X[:, -1, :2:][:, None, :] * 240.0, vel[:, :-1]], axis=1)
sp = np.linalg.norm(v_prev, axis=2, keepdims=True)
fric = v_prev * np.where(gr[:, None, None],
                         np.maximum(1.0 - np.maximum(sp, 100.0) * 4.0 * DT / np.maximum(sp, 1e-6), 0.0),
                         1.0)
dv = vel - fric
n = np.linalg.norm(dv, axis=2)  # |dv| per tick
dmg = KB[:, :, 2]
dr = np.hypot(KB[:, :, 0], KB[:, :, 1])
directed = (KB[:, :, 3] > 0.5) & (dr > 0.5) & (dmg > 0)
valid = directed & (Vd > 0) & np.isfinite(n)
print('%s%s directed+valid ticks: %d (%.4f)' % (tag, suf, valid.sum(),
      valid.sum() / valid.size), flush=True)
x = dmg[valid]
y = n[valid]
# least squares y = Kx + c
A = np.stack([x, np.ones_like(x)], axis=1)
K, c = np.linalg.lstsq(A, y, rcond=None)[0]
pred = K * x + c
ss = ((y - pred) ** 2).sum()
r2 = 1 - ss / ((y - y.mean()) ** 2).sum()
print('K_fit = %.4f hu/s per dmg, intercept %.1f hu/s, R2 = %.3f (n=%d)' % (
    K, c, r2, len(x)), flush=True)
for lo, hi in [(0, 30), (30, 60), (60, 120), (120, 1000)]:
    m = (x >= lo) & (x < hi)
    if m.sum() > 50:
        print('  dmg %3d-%4d: n=%6d median|dv|=%7.1f median|dv|/dmg=%6.2f' % (
            lo, hi, m.sum(), np.median(y[m]), np.median(y[m] / x[m])) , flush=True)
