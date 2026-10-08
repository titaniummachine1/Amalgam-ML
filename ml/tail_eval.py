"""Tail analysis: where does the model beat coast? High-curvature subsets."""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
ev = np.load(os.path.join(D, 'eval2.npz'), mmap_mode='r')
Xev, Yev, Wev, Hev = ev['X'], ev['Y'], ev['W'], ev['H']
mev = (np.abs(Yev).max(axis=1) <= 40)
Xev, Yev, Hev = Xev[mev], Yev[mev], Hev[mev]
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
net.load_state_dict(torch.load(os.path.join(D, 'gru64_v3.pt'), map_location='cpu'))
net.eval()

curv = np.abs(Yev).max(axis=1)  # per-second-normalized accel magnitude proxy
for q, name in [(0.0, 'all'), (50, 'p50'), (80, 'p80'), (90, 'p90'), (95, 'p95')]:
    thr = np.percentile(curv, q)
    m = curv >= thr
    Xs, Hs = Xev[m], Hev[m]
    # coast rollout
    v0 = Xs[:, -1, :2] * 240.0
    errs_c, errs_m = np.zeros(len(HOR)), np.zeros(len(HOR))
    # model rollout in chunks
    with torch.no_grad():
        for b in range(0, len(Xs), 256):
            xb = torch.from_numpy(Xs[b:b + 256])
            _, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * 240.0
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            hh = h
            hs = []
            for _ in range(66):
                a = net.head(hh[-1]) * 240.0
                v = v + a * DT
                p = p + v * DT
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / 240.0
                xin = torch.cat([v / 240.0, sp, gr, torch.zeros_like(sp)], dim=-1).unsqueeze(1)
                _, hh = net.gru(xin, hh)
                hs.append(p.clone())
            traj = torch.stack(hs, dim=1).numpy()
            for hi, h_ in enumerate(HOR):
                tgt = Hs[b:b + 256, hi]
                mm = ~np.isnan(tgt[:, 0])
                errs_m[hi] += np.abs(traj[mm, h_ - 1] - tgt[mm]).mean() * mm.sum()
    n = m.sum()
    errs_m /= n
    for hi, h_ in enumerate(HOR):
        pc = v0 * (h_ * DT)
        tgt = Hs[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        errs_c[hi] = np.abs(pc[mm] - tgt[mm]).mean()
    print(f'{name} (n={n}): coast={' '.join(f"{e:.1f}" for e in errs_c)} | model={' '.join(f"{e:.1f}" for e in errs_m)}',
          flush=True)
