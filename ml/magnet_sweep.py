"""Magnetic pull sweep (offline experiment, no retrain):
after each rollout accel prediction (ego frame), rotate its DIRECTION toward the
nearest 45-degree mode (WASD combos around yaw-forward) by strength s, magnitude
untouched. Net + training unaffected; output shaping only. Judged by hit-rate.
s=0 control must reproduce raw numbers."""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0
MAXV = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(14, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


net = Net()
net.load_state_dict(torch.load(os.path.join(D, 'gru_all_v5.pt'), map_location='cpu'))
net.eval()
ev = np.load(os.path.join(D, 'evalA.npz'), mmap_mode='r')
X, Y, H = np.array(ev['X']), np.array(ev['Y']), np.array(ev['H'])
m = (np.abs(Y).max(axis=1) <= 40)
X, H = X[m], H[m]
idx = np.random.RandomState(1).choice(len(X), min(2000, len(X)), replace=False)
X, H = X[idx], H[idx]
print('n =', len(X), flush=True)
mv_all = (X[:, -1, 5:] * MAXV).sum(axis=1, keepdims=True)


def rollout(Xs, strength):
    outs = []
    with torch.no_grad():
        for b in range(0, len(Xs), 128):
            xb = torch.from_numpy(Xs[b:b + 128])
            mv = torch.from_numpy((xb[:, -1, 5:].numpy() * MAXV).sum(-1, keepdims=True))
            _, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * mv
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            hh = h
            hs = []
            for _ in range(66):
                a = net.head(hh[-1]) * mv
                if strength > 0:
                    mag = a.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    th = torch.atan2(a[:, 1:2], a[:, 0:1])
                    snapped = torch.round(th / (np.pi / 4)) * (np.pi / 4)
                    d = (snapped - th + np.pi) % (2 * np.pi) - np.pi
                    th2 = th + strength * d
                    a = torch.cat([torch.cos(th2), torch.sin(th2)], dim=1) * mag
                v = v + a * DT
                p = p + v * DT
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / mv
                dyn = torch.cat([(v / mv).unsqueeze(1), sp.unsqueeze(1),
                                 gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
                _, hh = net.gru(torch.cat([dyn, xb[:, -1:, 5:]], dim=-1), hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


v0 = X[:, -1, :2] * (X[:, -1, 5:] * MAXV).sum(axis=1, keepdims=True)
names = ['100', '200', '300', '500', '1000']
for s in [0.0, 0.05, 0.15, 0.3, 0.5]:
    tr = rollout(X, s)
    line = []
    for hi, h_ in enumerate(HOR):
        tgt = H[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        err = np.abs(tr[mm][:, h_ - 1] - tgt[mm]).max(axis=1)
        line.append(f'{names[hi]}:{(err < HITR).mean() * 100:.1f}%')
    print(f's={s}:', ' '.join(line), flush=True)
