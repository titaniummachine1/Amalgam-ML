"""Hit-rate @24hu: GRU+attention (gru_both27.pt) vs champion-sim vs coast.
Same 2500-row RandomState(1) protocol as hit_eval_final.py.
Usage: python ml/hit_eval_both.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_both27 as T
from hit_eval_final import Net20, rollout20

torch.set_num_threads(4)
D = T.D
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0
MAXV = 240.0

d = np.load(os.path.join(D, 'evalB.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalBk.npz'))['K']
tw = np.load(os.path.join(D, 'evalB_threatwin.npz'), mmap_mode='r')
Xb = np.array(d['X'])
TW = np.array(tw['TW'])
TS = np.array(tw['TS'])[:, -1, :]
H = np.array(d['H'])
Y = np.array(d['Y'])
sol = np.where(k[:, 3] == 3)[0]
Xb, TW, TS = Xb[sol], TW, TS
W7 = np.array(np.load(os.path.join(D, 'evalB_w7.npz'))['W7'])
H, Y = H[sol], Y[sol]
m = (np.abs(Y).max(axis=1) <= 40)
Xb, TW, TS, H = Xb[m], TW[m], TS[m], H[m]
X20 = np.concatenate([Xb, TW], axis=2)
X27 = np.concatenate([X20, W7], axis=2)
active = (TW[:, -1, 3] > 0) | (TW[:, -1, 4] > 0)
print('n =', len(Xb), flush=True)

sim = Net20()
sim.load_state_dict(torch.load(os.path.join(D, 'gru_threatsimB.pt'), map_location='cpu'))
sim.eval()
att = T.Net27()
att.load_state_dict(torch.load(os.path.join(D, 'gru_both27.pt'), map_location='cpu'))
att.eval()

idx = np.random.RandomState(1).choice(len(Xb), min(2500, len(Xb)), replace=False)
Xb, X20, TS = Xb[idx], X20[idx], TS[idx]
X27 = X27[idx]
H = H[idx]
act = active[idx]
v0 = Xb[:, -1, :2] * MAXV
print('rolling sim...', flush=True)
trS = rollout20(sim, X20, TS, True)
print('rolling attn...', flush=True)
outs = []
with torch.no_grad():
    for b in range(0, len(Xb), 64):
        xb = torch.from_numpy(np.array(X27[b:b + 64]))
        sb = torch.from_numpy(np.array(TS[b:b + 64]))
        outs.append(T.unroll(att, xb, sb).numpy())
trA = np.concatenate(outs, axis=0)


def report(mask, label):
    if mask.sum() < 100:
        return
    print('--- %s (n=%d) ---' % (label, mask.sum()), flush=True)
    for name, arr in [('coast', None), ('sim', trS), ('attn', trA)]:
        line = []
        for hi, h_ in enumerate(HOR):
            tgt = H[mask][:, hi]
            mm = ~np.isnan(tgt[:, 0])
            if name == 'coast':
                err = np.abs((v0[mask] * (h_ * DT))[mm] - tgt[mm]).max(axis=1)
            else:
                err = np.abs(arr[mask][mm][:, h_ - 1] - tgt[mm]).max(axis=1)
            line.append('%d:%.1f%%' % (h_, (err < HITR).mean() * 100))
        print(' ', name, ' '.join(line), flush=True)


report(np.ones(len(Xb), dtype=bool), 'ALL')
report(act, 'THREAT-ACTIVE')
