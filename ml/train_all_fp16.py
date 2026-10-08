"""All-class training, FP16 mixed precision (torch.cpu.amp), fp32 master weights.
GRU(14,96). Weighted MSE 10 eps -> gru_all_fp16.pt. Export INT8 stays runtime."""
import os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
tr = np.load(os.path.join(D, 'trainA.npz'), mmap_mode='r')
ev = np.load(os.path.join(D, 'evalA.npz'), mmap_mode='r')
Xtr, Ytr, Wtr = tr['X'], tr['Y'], tr['W']
Xev, Yev, Wev = ev['X'], ev['Y'], ev['W']
mtr = (np.abs(Ytr).max(axis=1) <= 40)
mev = (np.abs(Yev).max(axis=1) <= 40)
Xtr, Ytr, Wtr = Xtr[mtr], Ytr[mtr], Wtr[mtr]
Xev, Yev, Wev = Xev[mev], Yev[mev], Wev[mev]
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(14, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


net = Net()
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
scaler = torch.amp.GradScaler('cpu')
lossf = nn.MSELoss(reduction='none')
B = 256
N = len(Xtr)
best = 1e18
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
        with torch.cpu.amp.autocast(dtype=torch.float16):
            l = (lossf(net(xb), yb) * wb).mean()
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
        tot += l.item() * len(idx)
    net.eval()
    with torch.no_grad():
        evl = 0.0
        nb = 0
        for i in range(0, len(Xev), 2048):
            xb = torch.from_numpy(np.array(Xev[i:i + 2048]))
            yb = torch.from_numpy(np.array(Yev[i:i + 2048]))
            wb = torch.from_numpy(np.array(Wev[i:i + 2048])).view(-1, 1)
            with torch.cpu.amp.autocast(dtype=torch.float16):
                evl += (lossf(net(xb), yb) * wb).mean() * len(xb)
            nb += len(xb)
        evl = float(evl / nb)
    print(f'ep{ep} train={tot / N:.4f} eval={evl:.4f} {time.time() - t0:.0f}s', flush=True)
    if evl < best:
        best = evl
        torch.save(net.state_dict(), os.path.join(D, 'gru_all_fp16.pt'))
print('best', best, flush=True)
