"""Standalone rollout eval from saved checkpoint. No retraining."""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
ev = np.load(os.path.join(D, 'eval2.npz'), mmap_mode='r')
Xev, Yev, Wev, Hev = ev['X'], ev['Y'], ev['W'], ev['H']
mev = (np.abs(Yev).max(axis=1) <= 40)
Xev, Hev = Xev[mev], Hev[mev]
print('eval', Xev.shape, flush=True)
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0


class Net(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.gru = nn.GRU(5, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


net = Net()
net.load_state_dict(torch.load(os.path.join(D, 'gru64_v2.pt'), map_location='cpu'))
net.eval()

idx = np.random.RandomState(0).choice(len(Xev), min(2000, len(Xev)), replace=False)
Xs, Hs = np.array(Xev[idx]), np.array(Hev[idx])
errs = np.zeros(len(HOR))
counts = np.zeros(len(HOR))
with torch.no_grad():
    for b in range(0, len(Xs), 128):
        xb = torch.from_numpy(Xs[b:b + 128])
        _, h = net.gru(xb)
        v = xb[:, -1, :2].clone() * 240.0
        p = torch.zeros_like(v)
        gr_frozen = xb[:, -1, 3:4]
        hh = h
        hs = []
        for _ in range(66):
            a = net.head(hh[-1]) * 240.0
            v = v + a * DT
            p = p + v * DT
            sp = (v ** 2).sum(-1, keepdim=True).sqrt() / 240.0
            xin = torch.cat([v / 240.0, sp, gr_frozen, torch.zeros_like(sp)], dim=-1).unsqueeze(1)
            _, hh = net.gru(xin, hh)
            hs.append(p.clone())
        traj = torch.stack(hs, dim=1).numpy()
        for hi, h_ in enumerate(HOR):
            tgt = Hs[b:b + 128, hi]
            m = ~np.isnan(tgt[:, 0])
            if m.sum():
                errs[hi] += np.abs(traj[m, h_ - 1] - tgt[m]).mean() * m.sum()
                counts[hi] += m.sum()
errs /= np.maximum(counts, 1)
print('model rollout pos err hu @100/200/300/500/1000ms:', ' '.join(f'{e:.1f}' for e in errs), flush=True)
