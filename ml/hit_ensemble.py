"""Ensemble test: champion-sim (threat) vs denseE (weapon) vs corrector.
Blends trajectories (no training). Same 2500-row protocol; rows drawn from
evalE (has weapon dims) and key-mapped to evalB/threatwin/dense132.
Usage: python ml/hit_ensemble.py"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hit_eval_final import Net20, rollout20
from train_correct132 import Corrector, build_tokens, corr_scaled

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0
MAXV = 240.0
MAXV1 = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)


class NetE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(21, 96, batch_first=True)
        self.head = nn.Linear(96, 2)


def apply_friction(v, gr):
    sp = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-12)
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


def rolloutE(net, X):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = torch.from_numpy(X[b:b + 128])
            mv = (xb[:, -1:, 5:14] * torch.from_numpy(MAXV1)).sum(-1, keepdim=True)
            _, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * mv.squeeze(1)
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            static = xb[:, -1:, 5:14]
            wstat = xb[:, -1:, 14:21]
            hh = h
            hs = []
            for _ in range(66):
                a = net.head(hh[-1]) * mv.squeeze(1)
                v = apply_friction(v, gr) + a * DT
                p = p + v * DT
                sp = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-12) / mv.squeeze(1)
                dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                                 gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
                _, hh = net.gru(torch.cat([dyn, static, wstat], dim=-1), hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


def rolloutC(net, X, S):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 256):
            xb = torch.from_numpy(X[b:b + 256])
            sb = torch.from_numpy(S[b:b + 256])
            pref, tok, coast = build_tokens(xb, sb)
            outs.append((coast + corr_scaled(net, pref, tok)).numpy())
    return np.concatenate(outs, axis=0)


# ---- data + key join ----
dB = np.load(os.path.join(D, 'evalB.npz'), mmap_mode='r')
kB = np.load(os.path.join(D, 'evalBk.npz'))['K']
dE = np.load(os.path.join(D, 'evalE.npz'), mmap_mode='r')
kE = np.load(os.path.join(D, 'evalEk.npz'))['K']
wE = np.load(os.path.join(D, 'evalE_wld.npz'), mmap_mode='r')['WLD']
tw = np.load(os.path.join(D, 'evalB_threatwin.npz'), mmap_mode='r')
mE = np.load(os.path.join(D, 'evalCmask.npz'))['keep']
Xb, Yb, HB = np.array(dB['X']), np.array(dB['Y']), np.array(dB['H'])
solB = np.where(kB[:, 3] == 3)[0]
posB = {tuple(kB[j][:3]): j for j in range(len(kB))}
posS = {tuple(kB[j][:3]): n for n, j in enumerate(solB)}
TW, TS = np.array(tw['TW']), np.array(tw['TS'])[:, -1, :]
XE = np.array(dE['X'])
goodE = np.where((kE[:, 3] == 3) & mE)[0]
rs = np.random.RandomState(1)
pick = rs.choice(goodE, min(2500, len(goodE)), replace=False)
rowsB, rowsS, rowsE, keep = [], [], [], []
for j in pick:
    key = tuple(kE[j][:3])
    if key in posB and key in posS:
        rowsB.append(posB[key])
        rowsS.append(posS[key])
        rowsE.append(j)
        keep.append(True)
rowsB, rowsS, rowsE = np.array(rowsB), np.array(rowsS), np.array(rowsE)
print('matched rows:', len(rowsB), flush=True)
Xb, Yb, HB = Xb[rowsB], Yb[rowsB], HB[rowsB]
X20 = np.concatenate([Xb, TW[rowsS]], axis=2)
S0 = TS[rowsS]
X21 = np.concatenate([XE[rowsE], np.array(wE[rowsE]).astype(np.float32) / 16.0], axis=2)
v0 = Xb[:, -1, :2] * MAXV
act = (TW[rowsS][:, -1, 3] > 0) | (TW[rowsS][:, -1, 4] > 0)

sim = Net20()
sim.load_state_dict(torch.load(os.path.join(D, 'gru_threatsimB.pt'), map_location='cpu'))
sim.eval()
den = NetE()
den.load_state_dict(torch.load(os.path.join(D, 'gru_soldier_denseE.pt'), map_location='cpu'))
den.eval()
cor = Corrector()
cor.load_state_dict(torch.load(os.path.join(D, 'correct132.pt'), map_location='cpu'))
cor.eval()

print('rolling sim...', flush=True)
trS = rollout20(sim, X20, S0, True)
print('rolling denseE...', flush=True)
trE = rolloutE(den, X21)
print('rolling corrector...', flush=True)
trC = rolloutC(cor, X20, S0)[:, :66]

blends = {
    'coast': None, 'sim': trS, 'denseE': trE, 'corr': trC,
    'sim+denseE': 0.5 * trS + 0.5 * trE,
    'sim+corr': 0.5 * trS + 0.5 * trC,
    'all3': (trS + trE + trC) / 3.0,
}


def report(mask, label):
    if mask.sum() < 100:
        return
    print('--- %s (n=%d) ---' % (label, mask.sum()), flush=True)
    for name, arr in blends.items():
        line = []
        for hi, h_ in enumerate(HOR):
            tgt = HB[mask][:, hi]
            mm = ~np.isnan(tgt[:, 0])
            if arr is None:
                err = np.abs((v0[mask] * (h_ * DT))[mm] - tgt[mm]).max(axis=1)
            else:
                err = np.abs(arr[mask][mm][:, h_ - 1] - tgt[mm]).max(axis=1)
            line.append('%d:%.1f%%' % (h_, (err < HITR).mean() * 100))
        print(' ', name, ' '.join(line), flush=True)


report(np.ones(len(Xb), dtype=bool), 'ALL')
report(act, 'THREAT-ACTIVE')
