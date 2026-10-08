"""Warmup-length ablation on FROZEN v4 (soldier): does hit-rate need 1s, or is
0.5s/0.33s enough? Feeds last-K ticks of the 66 warmup, rolls out, hit-rate.
Answers: single short buffer optimal length. -> warmup_ablation.log"""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(5, 64, batch_first=True)
        self.head = nn.Linear(64, 2)


net = Net()
net.load_state_dict(torch.load(os.path.join(D, 'gru64_v4.pt'), map_location='cpu'))
net.eval()
d = np.load(os.path.join(D, 'eval3.npz'), mmap_mode='r')
X, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
m = (np.abs(Y).max(axis=1) <= 40)
X, H = X[m], H[m]
idx = np.random.RandomState(1).choice(len(X), min(1500, len(X)), replace=False)
X, H = X[idx], H[idx]
print('n =', len(X), flush=True)


def rollout(Xw):
    outs = []
    with torch.no_grad():
        for b in range(0, len(Xw), 128):
            xb = torch.from_numpy(Xw[b:b + 128])
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
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


for K in [22, 33, 66]:
    tr = rollout(X[:, -K:, :])
    line = []
    for hi, h_ in enumerate(HOR):
        tgt = H[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        err = np.abs(tr[mm][:, h_ - 1] - tgt[mm]).max(axis=1)
        line.append(f'{h_}:{(err < HITR).mean() * 100:.1f}%')
    print(f'warmup={K} ({K / 66:.2f}s):', ' '.join(line), flush=True)
