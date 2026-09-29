"""Raw+ctx vs residual+ctx: same net, same data, different target.
raw: Y = pure-input accel. residual: Y = pure-input accel - heur (net predicts
correction over yaw-rate heuristic; inference adds heur back).
Weighted MSE, 10 eps, GRU-80, input 37. Usage: train_ctx.py [raw|res]."""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
MODE = sys.argv[1] if len(sys.argv) > 1 else 'raw'


def load(tag):
    d = np.load(os.path.join(D, tag + '3.npz'), mmap_mode='r')
    e = np.load(os.path.join(D, tag + '_ctx.npz'), mmap_mode='r')
    Xb = np.array(d['X'])
    Es = np.repeat(np.array(e['E'])[:, None, :], Xb.shape[1], axis=1)
    X = np.concatenate([Xb, Es], axis=2)  # (N,66,37)
    Y = np.array(d['Y'])
    H = np.array(e['E'])[:, 27:29]  # heur in same units
    return X, Y, np.array(d['W']), H


Xtr, Ytr, Wtr, Htr = load('train')
Xev, Yev, Wev, Hev = load('eval')
mtr = (np.abs(Ytr).max(axis=1) <= 40)
mev = (np.abs(Yev).max(axis=1) <= 40)
Xtr, Ytr, Wtr, Htr = Xtr[mtr], Ytr[mtr], Wtr[mtr], Htr[mtr]
Xev, Yev, Wev, Hev = Xev[mev], Yev[mev], Wev[mev], Hev[mev]
if MODE == 'res':
    Ytr = Ytr - Htr
    Yev = Yev - Hev
print(MODE, 'train', Xtr.shape, 'eval', Xev.shape, flush=True)


class Net(nn.Module):
    def __init__(self, h=80):
        super().__init__()
        self.gru = nn.GRU(37, h, batch_first=True)
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
tag = 'rawctx' if MODE == 'raw' else 'resctx'
for ep in range(10):
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
    print(f'{tag} ep{ep} train={tot / N:.4f} eval={evl:.4f} {time.time() - t0:.0f}s', flush=True)
    if evl < best:
        best = evl
        torch.save(net.state_dict(), os.path.join(D, tag + '.pt'))
print(tag, 'best', best, flush=True)
