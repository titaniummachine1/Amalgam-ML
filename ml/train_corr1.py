"""Retrain GRU with correction-history inputs (5 base + 24 corr). Same weighted
loss, baselines, checkpoint + rollout report as train3."""
import os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')


def load(tag):
    d = np.load(os.path.join(D, tag + '3.npz'), mmap_mode='r')
    c = np.load(os.path.join(D, tag + '_corr1.npz'), mmap_mode='r')
    Xb = np.array(d['X'])
    Cb = np.array(c['C'])
    Cb[:, 0::4] /= 50.0
    Cb[:, 1::4] /= 50.0  # err hu -> ~unit scale (predacc already /240)
    Cb = np.repeat(Cb[:, None, :], Xb.shape[1], axis=1)
    X = np.concatenate([Xb, Cb], axis=2)
    return X, np.array(d['Y']), np.array(d['W']), np.array(d['H'])


Xtr, Ytr, Wtr, Htr = load('train')
Xev, Yev, Wev, Hev = load('eval')
mtr = (np.abs(Ytr).max(axis=1) <= 40)
mev = (np.abs(Yev).max(axis=1) <= 40)
Xtr, Ytr, Wtr = Xtr[mtr], Ytr[mtr], Wtr[mtr]
Xev, Yev, Wev = Xev[mev], Yev[mev], Wev[mev]
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0


class Net(nn.Module):
    def __init__(self, h=80):
        super().__init__()
        self.gru = nn.GRU(29, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


net = Net()
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
lossf = nn.MSELoss(reduction='none')
B = 256
N = len(Xtr)
best = 1e18
for ep in range(8):
    net.train()
    perm = np.random.permutation(N)
    tot = 0.0
    t0 = time.time()
    for i in range(0, N, B):
        idx = perm[i:i + B]
        xb = torch.from_numpy(np.array(Xtr[idx]))
        yb = torch.from_numpy(np.array(Ytr[idx]))
        wb = torch.from_numpy(np.array(Wtr[idx])).view(-1, 1)
        opt.zero_grad()
        l = (lossf(net(xb), yb) * wb).mean()
        l.backward()
        opt.step()
        tot += l.item() * len(idx)
    net.eval()
    with torch.no_grad():
        evl = 0.0
        nb = 0
        for i in range(0, len(Xev), 2048):
            xb = torch.from_numpy(np.array(Xev[i:i + 2048]))
            yb = torch.from_numpy(np.array(Yev[i:i + 2048]))
            wb = torch.from_numpy(np.array(Wev[i:i + 2048])).view(-1, 1)
            evl += (lossf(net(xb), yb) * wb).mean() * len(xb)
            nb += len(xb)
        evl = float(evl / nb)
    print(f'ep{ep} train_wMSE={tot / N:.4f} eval_wMSE={evl:.4f} {time.time() - t0:.0f}s', flush=True)
    if evl < best:
        best = evl
        torch.save(net.state_dict(), os.path.join(D, 'gru_corr1.pt'))
print('best', best, flush=True)
