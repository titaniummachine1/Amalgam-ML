"""Verdict for the known-forces accel policy. Same protocol as hit_eval_final.py
(evalB soldier, |Y|<=40 mask, RandomState(1) 2500 rows): coast, B-champion
(gru_threatsimB.pt), forces model. The forces net is 29-dim; champion rows use
the 20-dim accel rollout for reference.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inputspace as P

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
HITR = 24.0
MAXV = 240.0
CK = os.environ.get('CKPT', 'gru_forcesA.pt')
K_IMP = float(os.environ.get('K_IMP', '3.0'))

d = np.load(os.path.join(D, 'evalB.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalBk.npz'))['K']
tw = np.load(os.path.join(D, 'evalB_threatwin.npz'))
kb = np.load(os.path.join(D, 'evalB_knock.npz'))
ts = np.load(os.path.join(D, 'evalB_tsf.npz'))
Xb, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
TW, TS = np.array(tw['TW']), np.array(tw['TS'])
sol = k[:, 3] == 3
Xb, TW, TS = Xb[sol], TW, TS
KBW = np.array(kb['KBW'])
TSF = np.array(ts['TSF'])
X29 = np.concatenate([Xb, TW, KBW, TSF], axis=2)
X20 = np.concatenate([Xb, TW], axis=2)
S0 = TS[:, -1, :]
m = (np.abs(Y[sol]).max(axis=1) <= 40)
Xb, X20, X29, S0 = Xb[m], X20[m], X29[m], S0[m]
H = H[sol][m]
KB = np.array(kb['KB'])[m]
active = (TW[m][:, -1, 3] > 0) | (TW[m][:, -1, 4] > 0)

idx = np.random.RandomState(1).choice(len(Xb), min(2500, len(Xb)), replace=False)
Xb, X20, X29, S0, H, KB, active = (a[idx] for a in (Xb, X20, X29, S0, H, KB, active))
Y = Y[sol][m][idx]
v0 = Xb[:, -1, :2] * MAXV
print('n =', len(Xb), 'threat-active frac = %.3f' % active.mean(), flush=True)


def roll_accel(net, X, S):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            outs.append(P.rollout(net, torch.from_numpy(X[b:b + 128]),
                                  torch.from_numpy(S[b:b + 128]),
                                  mode='accel').numpy())
    return np.concatenate(outs, axis=0)


def roll_forces(net, X, S, K):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            outs.append(P.rollout_forces(
                net, torch.from_numpy(X[b:b + 128]), torch.from_numpy(S[b:b + 128]),
                torch.from_numpy(K[b:b + 128]), k_imp=K_IMP).numpy())
    return np.concatenate(outs, axis=0)


def report(mask, label, rows):
    if mask.sum() < 100:
        return
    print('--- %s (n=%d) ---' % (label, mask.sum()), flush=True)
    for name, arr in rows:
        line = []
        for hi, h_ in enumerate(HOR):
            tgt = H[mask][:, hi]
            mm = ~np.isnan(tgt[:, 0])
            err = (np.abs((v0[mask] * (h_ * P.DT))[mm] - tgt[mm]).max(axis=1)
                   if arr is None else
                   np.abs(arr[mask][mm][:, h_ - 1] - tgt[mm]).max(axis=1))
            line.append('%d:%.1f%%' % (h_, (err < HITR).mean() * 100))
        print('  %-9s %s' % (name, ' '.join(line)), flush=True)


rows = [('coast', None)]
champ = P.Net()
champ.load_state_dict(torch.load(os.path.join(D, 'gru_threatsimB.pt'), map_location='cpu'))
champ.eval()
print('rolling B accel champion...', flush=True)
rows.append(('champ', roll_accel(champ, X20, S0)))
if os.path.exists(os.path.join(D, CK)):
    net = P.Net(n_in=20 + P.NEX, n_out=2)
    net.load_state_dict(torch.load(os.path.join(D, CK), map_location='cpu'))
    net.eval()
    print('rolling %s...' % CK, flush=True)
    rows.append(('forces', roll_forces(net, X29, S0, KB)))
else:
    print('(no %s yet)' % CK, flush=True)

mask_all = np.ones(len(Xb), dtype=bool)
report(mask_all, 'ALL', rows)
report(active, 'THREAT-ACTIVE', rows)
has = (KB[:, :, 3].sum(axis=1) > 0.5)
if has.sum() >= 100:
    report(has, 'KNOCKBACK', rows)
