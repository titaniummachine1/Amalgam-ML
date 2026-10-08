"""D-proof: coast vs denseC vs denseD (+air-blend variants) on identical rows.

B-line (14-dim accel, friction unroll = C++ runtime math), evalB soldier +
combat mask, FULL masked set (no subsample), regime slices:
  ALL / GROUND / AIR (gr@T==0) / AIR+FAST (speed@T>350, dive proxy)
Metric: hit-rate @24hu at 7/13/20/33/66 vs H horizons.
Blend: policy->coast ramp [20,33] gated on airborne@T (the C++ air fix).
Ship rule: D replaces C only if wins-or-tied everywhere with no slice
regression >=1pp; else C+blend stays.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0
MAXV = 240.0


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(14, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def apply_friction(v, gr):
    sp = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-12)
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


def rollout(net, X):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = torch.from_numpy(X[b:b + 128])
            _, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * MAXV
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            hh = h
            hs = []
            for _ in range(66):
                a = net.head(hh[-1]) * MAXV
                v = apply_friction(v, gr) + a * DT
                p = p + v * DT
                sp = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-12) / MAXV
                dyn = torch.cat([(v / MAXV).unsqueeze(1), sp.unsqueeze(1),
                                 gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
                _, hh = net.gru(torch.cat([dyn, xb[:, -1:, 5:]], dim=-1), hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


d = np.load(os.path.join(D, 'evalB.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalBk.npz'))['K']
X, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
mk = np.load(os.path.join(D, 'evalCmask.npz'))['keep']
m = (k[:, 3] == 3) & mk
X, H = X[m], H[m]
gr0 = X[:, -1, 3]
sp0 = np.linalg.norm(X[:, -1, :2], axis=1) * MAXV
v0 = X[:, -1, :2] * MAXV
air = gr0 < 0.5
print('n =', len(X), 'air frac = %.3f fast-air frac = %.3f' % (
    air.mean(), ((air) & (sp0 > 350)).mean()), flush=True)


def coast_traj():
    steps = np.arange(1, 67, dtype=np.float32) * DT
    return v0[:, None, :] * steps[None, :, None]


def blend_row(pol, a=20, b=33):
    ramp = np.clip((np.arange(1, 67, dtype=np.float32) - a) / max(b - a, 1), 0, 1)
    w = np.where(air[:, None], ramp[None, :], 0.0)
    return (1 - w[:, :, None]) * pol + w[:, :, None] * coast_traj()


rows = []
_C = coast_traj()
for ck, tag in [('gru_soldier_denseC.pt', 'C'), ('gru_soldier_denseD.pt', 'D')]:
    p = os.path.join(D, ck)
    if not os.path.exists(p):
        print('(no %s)' % ck, flush=True)
        continue
    net = Net()
    net.load_state_dict(torch.load(p, map_location='cpu'))
    net.eval()
    print('rolling %s...' % ck, flush=True)
    pol = rollout(net, X)
    rows.append((tag, pol))
    rows.append((tag + '+blend', blend_row(pol)))


def report(mask, label):
    if mask.sum() < 100:
        print('--- %s SKIPPED (n=%d) ---' % (label, mask.sum()), flush=True)
        return
    print('--- %s (n=%d) ---' % (label, mask.sum()), flush=True)
    for name, arr in [('coast', None)] + rows:
        line = []
        for hi, h_ in enumerate(HOR):
            tgt = H[mask][:, hi]
            mm = ~np.isnan(tgt[:, 0])
            err = (np.abs((v0[mask] * (h_ * DT))[mm] - tgt[mm]).max(axis=1)
                   if arr is None else
                   np.abs(arr[mask][mm][:, h_ - 1] - tgt[mm]).max(axis=1))
            line.append('%d:%.1f%%' % (h_, (err < HITR).mean() * 100))
        print('  %-10s %s' % (name, ' '.join(line)), flush=True)


for mask, lab in [(np.ones(len(X), dtype=bool), 'ALL'),
                  (gr0 > 0.5, 'GROUND'),
                  (air, 'AIR'),
                  (air & (sp0 > 350), 'AIR+FAST')]:
    report(mask, lab)
