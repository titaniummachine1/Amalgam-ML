"""Extended verdict @24hu to 2s: coast vs champion-sim(132) vs correct132.
Same 2500-row RandomState(1) protocol as hit_eval_final.py, horizons
[7,13,20,33,66,99,132] (~100ms..2s). Gracefully skips corrector if untrained.
Usage: python ml/hit_eval132.py"""
import os

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66, 99, 132]
NSTEPS = 132
DT = 1 / 66.0
HITR = 24.0
MAXV = 240.0


class Net20(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(20, 96, batch_first=True)
        self.head = nn.Linear(96, 2)


def apply_friction(v, gr):
    sp = (v ** 2).sum(-1, keepdim=True).sqrt()
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


d = np.load(os.path.join(D, 'evalB.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalBk.npz'))['K']
tw = np.load(os.path.join(D, 'evalB_threatwin.npz'))
dn = np.load(os.path.join(D, 'evalB_dense132.npz'))
Xb = np.array(d['X'])
TW, TS = np.array(tw['TW']), np.array(tw['TS'])
Hd, Vd = np.array(dn['Hd']), np.array(dn['Vd'])
sol = k[:, 3] == 3
Xb, TW, TS = Xb[sol], TW, TS
Hd, Vd = Hd, Vd  # already soldier-only, aligned
Y = np.array(d['Y'])[sol]
m = (np.abs(Y).max(axis=1) <= 40)
Xb, TW, TS, Hd, Vd = Xb[m], TW[m], TS[m], Hd[m], Vd[m]
X20 = np.concatenate([Xb, TW], axis=2)
S0 = TS[:, -1, :]
active = (TW[:, -1, 3] > 0) | (TW[:, -1, 4] > 0)
print('n =', len(Xb), 'threat-active frac = %.3f' % active.mean(), flush=True)

sim = Net20()
sim.load_state_dict(torch.load(os.path.join(D, 'gru_threatsimB.pt'), map_location='cpu'))
sim.eval()

CKPT = os.path.join(D, 'correct132.pt')
corr = None
if os.path.exists(CKPT):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_correct132 import Corrector, build_tokens, corr_scaled
    corr = Corrector()
    corr.load_state_dict(torch.load(CKPT, map_location='cpu'))
    corr.eval()
    print('corrector loaded', flush=True)
else:
    print('no corrector ckpt yet', flush=True)


def rollout_sim(net, X, S):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = torch.from_numpy(X[b:b + 128])
            sb = torch.from_numpy(S[b:b + 128])
            _, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * MAXV
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            static = xb[:, -1:, 5:14]
            thr_frozen = xb[:, -1:, 14:20]
            q = sb[:, 0:2].clone()
            vth = sb[:, 2:4].clone()
            present = sb[:, 4:5]
            nspd = torch.clamp((vth ** 2).sum(-1, keepdim=True).sqrt() / 3000.0, 0.0, 1.0)
            cnt = thr_frozen[:, 0, 3:4]
            mem = thr_frozen[:, 0, 4:5]
            hh = h
            hs = []
            for _ in range(NSTEPS):
                a = net.head(hh[-1]) * MAXV
                v = apply_friction(v, gr) + a * DT
                p = p + v * DT
                q = q + vth * DT
                rel = q - p
                dist = rel.norm(dim=1, keepdim=True).clamp_min(1e-6)
                ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
                ang = torch.atan2(rel[:, 1:2], rel[:, 0:1])
                infov = (ang.abs() < (np.pi / 4)).float()
                appr = -((vth - v) * rel).sum(-1, keepdim=True) / dist
                tti = torch.where(appr > 50.0, torch.clamp(dist / appr / 5.0, 0.0, 1.0),
                                  torch.ones_like(dist))
                live = torch.cat([ndist, nspd, infov, cnt, mem, tti], dim=-1).unsqueeze(1)
                thr = present.unsqueeze(1) * live + (1.0 - present.unsqueeze(1)) * thr_frozen
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / MAXV
                dyn = torch.cat([(v / MAXV).unsqueeze(1), sp.unsqueeze(1),
                                 gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1),
                                 thr], dim=-1)
                _, hh = net.gru(torch.cat([dyn, static], dim=-1), hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


def rollout_corr(net, X, S):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 256):
            xb = torch.from_numpy(X[b:b + 256])
            sb = torch.from_numpy(S[b:b + 256])
            pref, tok, coast = build_tokens(xb, sb)
            outs.append((coast + corr_scaled(net, pref, tok)).numpy())
    return np.concatenate(outs, axis=0)


idx = np.random.RandomState(1).choice(len(Xb), min(2500, len(Xb)), replace=False)
Xb, X20, S0 = Xb[idx], X20[idx], S0[idx]
Hd, Vd = Hd[idx], Vd[idx]
act = active[idx]
v0 = Xb[:, -1, :2] * MAXV
print('rolling sim...', flush=True)
trS = rollout_sim(sim, X20, S0)
trC = None
if corr is not None:
    print('rolling corrector...', flush=True)
    trC = rollout_corr(corr, X20, S0)


def report(mask, label):
    if mask.sum() < 100:
        return
    print('--- %s (n=%d) ---' % (label, mask.sum()), flush=True)
    arrs = [('coast', None), ('sim', trS)] + ([('corr', trC)] if trC is not None else [])
    for name, arr in arrs:
        line = []
        for h_ in HOR:
            tgt = Hd[mask][:, h_ - 1]
            vv = Vd[mask][:, h_ - 1]
            mm = vv & ~np.isnan(tgt[:, 0])
            if name == 'coast':
                err = np.abs((v0[mask] * (h_ * DT))[mm] - tgt[mm]).max(axis=1)
            else:
                err = np.abs(arr[mask][mm][:, h_ - 1] - tgt[mm]).max(axis=1)
            line.append('%d:%.1f%%' % (h_, (err < HITR).mean() * 100))
        print(' ', name, ' '.join(line), flush=True)


report(np.ones(len(Xb), dtype=bool), 'ALL')
report(act, 'THREAT-ACTIVE')
