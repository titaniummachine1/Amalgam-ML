"""Physics validation for the known-forces rollout (gates training).

Runs rollout_forces with ZERO policy accel (head outputs forced to 0) on evalB:
pure physics (friction + known impulses K=3.0 + collide hook identity) vs truth.
Reports per-tick position error overall + on knockback rows (has=1) + clean rows,
vs friction-coast and vs frictionless coast. Acceptance:
  1. physics+impulses beats friction-coast on has=1 rows (impulses explain motion)
  2. residual (truth - physics) is input-shaped: bounded by accel ceiling, else flag
  3. has=0 rows: physics == friction-coast exactly (sanity)
If (1) fails, K_imp/directions are wrong - fix forces, do NOT train.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inputspace as P

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DT = 1 / 66.0
K_IMP = float(os.environ.get('K_IMP', '3.0'))

d = np.load(os.path.join(D, 'evalB.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalBk.npz'))['K']
tw = np.load(os.path.join(D, 'evalB_threatwin.npz'))
dn = np.load(os.path.join(D, 'evalB_dense66.npz'))
kb = np.load(os.path.join(D, 'evalB_knock.npz'))
ts = np.load(os.path.join(D, 'evalB_tsf.npz'))
sol = k[:, 3] == 3
Xb = np.array(d['X'])[sol]
X = np.concatenate([Xb, np.array(tw['TW']), np.array(kb['KBW']),
                    np.array(ts['TSF'])], axis=2)
S0 = np.array(tw['TS'])[:, -1, :]
Hd, Vd = np.array(dn['Hd']), np.array(dn['Vd'])
KB = np.array(kb['KB'])
print('X', X.shape, 'KB', KB.shape, flush=True)

mv = 240.0
v0 = X[:, -1, :2] * mv
gr = X[:, -1, 3:4]
errs = {'phys': [], 'fric': []}
HOR = [7, 13, 20, 33, 66]
per_h = {h: {'phys': [], 'fric': [], 'has': []} for h in HOR}
has_any = (KB[:, :, 3].sum(axis=1) > 0.5)
print('rows with any future hurt: %.4f' % has_any.mean(), flush=True)
N = min(len(X), 20000)
idx = np.random.RandomState(7).choice(len(X), N, replace=False)
has_sub = has_any[idx]
for b in range(0, N, 512):
    j = idx[b:b + 512]
    hb = has_sub[b:b + 512]
    vv = torch.from_numpy(v0[j])
    pp, pf = torch.zeros_like(vv), torch.zeros_like(vv)
    vf, vc = vv.clone(), vv.clone()
    g = torch.from_numpy(gr[j])
    Tp, Tf = [], []
    for h in range(66):
        # physics + impulses
        sp = (vf ** 2).sum(-1, keepdim=True).sqrt()
        drop = torch.clamp(sp, min=100.0) * 4.0 * DT
        ns = torch.clamp(sp - drop, min=0.0)
        vf = vf * torch.where(sp > 0, ns / sp.clamp_min(1e-6), torch.ones_like(sp)) * g + vf * (1 - g)
        imp = torch.from_numpy(KB[j][:, h, 0:2].astype(np.float64)) * (torch.from_numpy(KB[j][:, h, 2:3].astype(np.float64)) * K_IMP)
        vf = (vf.double() + imp).float()
        pp = pp + vf * DT
        Tp.append(pp.clone())
        # friction only
        sp2 = (vc ** 2).sum(-1, keepdim=True).sqrt()
        drop2 = torch.clamp(sp2, min=100.0) * 4.0 * DT
        ns2 = torch.clamp(sp2 - drop2, min=0.0)
        vc = vc * torch.where(sp2 > 0, ns2 / sp2.clamp_min(1e-6), torch.ones_like(sp2)) * g + vc * (1 - g)
        pf = pf + vc * DT
        Tf.append(pf.clone())
    pp, pf = torch.stack(Tp, dim=1).numpy(), torch.stack(Tf, dim=1).numpy()
    tgt = Hd[j]
    m = Vd[j].astype(bool)
    for hi, h in enumerate(HOR):
        mm = m[:, h - 1]
        if mm.sum() == 0:
            continue
        # careful: Hd columns are 0-indexed ticks, HOR h -> column h-1
        ep = np.abs(pp[mm][:, h - 1] - tgt[mm][:, h - 1]).max(axis=1)
        ef = np.abs(pf[mm][:, h - 1] - tgt[mm][:, h - 1]).max(axis=1)
        per_h[h]['phys'].append(ep)
        per_h[h]['fric'].append(ef)
        per_h[h]['has'].append(hb[mm])
print('--- mean position error (hu) physics+impulses vs friction-only ---')
KBhas = (KB[:, :, 3].sum(axis=1) > 0.5)
for h in HOR:
    ep = np.concatenate(per_h[h]['phys'])
    ef = np.concatenate(per_h[h]['fric'])
    hh = np.concatenate(per_h[h]['has'])
    print('h=%2d ALL   phys %7.1f  fric %7.1f  win %+6.1f' % (h, ep.mean(), ef.mean(), ef.mean() - ep.mean()), flush=True)
    if hh.sum() > 50:
        print('h=%2d HURT  phys %7.1f  fric %7.1f  win %+6.1f  (n=%d)' % (
            h, ep[hh].mean(), ef[hh].mean(), ef[hh].mean() - ep[hh].mean(), hh.sum()), flush=True)
print('--- knockback rows only (any future hurt) ---', flush=True)
