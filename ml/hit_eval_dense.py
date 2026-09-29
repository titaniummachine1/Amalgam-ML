"""Hit-rate @24hu: dense66 soldier model vs coast. Proof of better-than-coast."""
import os
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


net = Net()
net.load_state_dict(torch.load(os.path.join(D, 'gru_soldier_dense66.pt'), map_location='cpu'))
net.eval()
d = np.load(os.path.join(D, 'evalA.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalAk.npz'))['K']
X, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
m = (k[:, 3] == 3) & (np.abs(Y).max(axis=1) <= 40)
X, H = X[m], H[m]
idx = np.random.RandomState(1).choice(len(X), min(3000, len(X)), replace=False)
X, H = X[idx], H[idx]
print('n =', len(X), flush=True)
v0 = X[:, -1, :2] * MAXV
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
            v = v + a * DT
            p = p + v * DT
            sp = (v ** 2).sum(-1, keepdim=True).sqrt() / MAXV
            dyn = torch.cat([(v / MAXV).unsqueeze(1), sp.unsqueeze(1),
                             gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
            _, hh = net.gru(torch.cat([dyn, xb[:, -1:, 5:]], dim=-1), hh)
            hs.append(p.clone())
        outs.append(torch.stack(hs, dim=1).numpy())
tr = np.concatenate(outs, axis=0)
for label, arr in [('coast', None), ('dense66', tr)]:
    line = []
    for hi, h_ in enumerate(HOR):
        tgt = H[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        err = (np.abs((v0 * (h_ * DT))[mm] - tgt[mm]).max(axis=1) if label == 'coast'
               else np.abs(arr[mm][:, h_ - 1] - tgt[mm]).max(axis=1))
        line.append(f'{h_}:{(err < HITR).mean() * 100:.1f}%')
    print(label, 'hit-rate@24hu', ' '.join(line), flush=True)
