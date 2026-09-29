"""Hit-rate eval: fraction of rollouts with position error < 24hu (half hitbox)
per horizon. coast vs v3ii (5-dim) vs corr (29-dim)."""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0


class Net5(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(5, 64, batch_first=True)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class Net29(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(29, 80, batch_first=True)
        self.head = nn.Linear(80, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


d = np.load(os.path.join(D, 'eval3.npz'), mmap_mode='r')
c = np.load(os.path.join(D, 'eval_corr1.npz'), mmap_mode='r')
Xb, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
Cb = np.array(c['C'])
Cb[:, 0::4] /= 50.0
Cb[:, 1::4] /= 50.0
m = (np.abs(Y).max(axis=1) <= 40)
Xb, Y, H, Cb = Xb[m], Y[m], H[m], Cb[m]
Xc = np.concatenate([Xb, np.repeat(Cb[:, None, :], 66, axis=1)], axis=2)
print('n =', len(Xb), flush=True)

net5 = Net5()
net5.load_state_dict(torch.load(os.path.join(D, 'gru64_v3ii.pt'), map_location='cpu'))
net5.eval()
net29 = Net29()
net29.load_state_dict(torch.load(os.path.join(D, 'gru_corr1.pt'), map_location='cpu'))
net29.eval()

idx = np.random.RandomState(1).choice(len(Xb), min(2000, len(Xb)), replace=False)
Xb, Xc, H = Xb[idx], Xc[idx], H[idx]
v0 = Xb[:, -1, :2] * 240.0

# coast trajectories per horizon
coast = {}
for hi, h_ in enumerate(HOR):
    coast[hi] = v0 * (h_ * DT)

# model rollouts
def rollout(net, X, scale_gru_in=5):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = torch.from_numpy(X[b:b + 128])
            _, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * 240.0
            p = torch.zeros_like(v)
            if scale_gru_in == 29:
                static = xb[:, -1:, 5:]
                gr = xb[:, -1, 3:4]
            else:
                gr = xb[:, -1, 3:4]
            hh = h
            hs = []
            for _ in range(66):
                a = net.head(hh[-1]) * 240.0
                v = v + a * DT
                p = p + v * DT
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / 240.0
                if scale_gru_in == 29:
                    xin = torch.cat([v / 240.0, sp, gr, torch.zeros_like(sp), static[:, 0, :]], dim=-1).unsqueeze(1)
                else:
                    xin = torch.cat([v / 240.0, sp, gr, torch.zeros_like(sp)], dim=-1).unsqueeze(1)
                _, hh = net.gru(xin, hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


print('rolling v3ii...', flush=True)
tr5 = rollout(net5, Xb, 5)
print('rolling corr...', flush=True)
tr29 = rollout(net29, Xc, 29)
names = ['100', '200', '300', '500', '1000']
for label, arr in [('coast', None), ('v3ii', tr5), ('corr', tr29)]:
    line = []
    for hi, h_ in enumerate(HOR):
        tgt = H[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        if label == 'coast':
            err = np.abs(coast[hi][mm] - tgt[mm]).max(axis=1)
        else:
            err = np.abs(arr[mm][:, h_ - 1] - tgt[mm]).max(axis=1)
        line.append(f'{names[hi]}:{(err < HITR).mean() * 100:.1f}%')
    print(label, 'hit-rate@24hu', ' '.join(line), flush=True)
