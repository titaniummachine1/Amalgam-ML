"""Phase 2: finetune GRU with differentiable multi-step rollout loss (position
error at horizons) + small single-step anchor. Fixes teacher-forcing exposure bias."""
import os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
tr = np.load(os.path.join(D, 'train2.npz'), mmap_mode='r')
ev = np.load(os.path.join(D, 'eval2.npz'), mmap_mode='r')
Xtr, Ytr, Wtr, Htr = tr['X'], tr['Y'], tr['W'], tr['H']
Xev, Hev = ev['X'], ev['H']
mtr = (np.abs(Ytr).max(axis=1) <= 40)
mev = (np.abs(ev['Y']).max(axis=1) <= 40)
Xtr, Ytr, Wtr, Htr = Xtr[mtr], Ytr[mtr], Wtr[mtr], Htr[mtr]
Xev, Hev = Xev[mev], Hev[mev]
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)
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


def unroll(net, xb):
    """Returns (B,66,2) ego-hu trajectory + single-step accel."""
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * 240.0
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    hh = h
    traj = []
    for _ in range(66):
        a = net.head(hh[-1]) * 240.0
        v = v + a * DT
        p = p + v * DT
        sp = (v ** 2).sum(-1, keepdim=True).sqrt() / 240.0
        xin = torch.cat([v / 240.0, sp, gr, torch.zeros_like(sp)], dim=-1).unsqueeze(1)
        _, hh = net.gru(xin, hh)
        traj.append(p)
    return torch.stack(traj, dim=1)


def pos_loss(net, xb, Hb):
    traj = unroll(net, xb)
    l = 0.0
    for hi, h_ in enumerate(HOR):
        tgt = torch.from_numpy(np.array(Hb[:, hi]))
        m = ~torch.isnan(tgt[:, 0:1])
        d = ((traj[:, h_ - 1] - tgt) ** 2).sum(-1, keepdim=True)
        l = l + (d * m).sum() / m.sum().clamp_min(1)
    return l / len(HOR)


net = Net()
net.load_state_dict(torch.load(os.path.join(D, 'gru64_v2.pt'), map_location='cpu'))
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
B = 128
N = len(Xtr)
best = 1e18
for ep in range(3):
    net.train()
    perm = np.random.permutation(N)
    tot = 0.0
    t0 = time.time()
    for i in range(0, N, B):
        idx = perm[i:i + B]
        xb = torch.from_numpy(np.array(Xtr[idx]))
        Hb = Htr[idx]
        opt.zero_grad()
        l = pos_loss(net, xb, Hb)
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot += l.item() * len(idx)
    # eval
    net.eval()
    with torch.no_grad():
        evl = 0.0; nb = 0
        for i in range(0, len(Xev), 512):
            xb = torch.from_numpy(np.array(Xev[i:i + 512]))
            Hb = Hev[i:i + 512]
            evl += pos_loss(net, xb, Hb).item() * len(xb)
            nb += len(xb)
        evl /= nb
    print(f'ep{ep} train_pos={tot / N:.2f}hu2 eval_pos={evl:.2f}hu2 {time.time() - t0:.0f}s', flush=True)
    if evl < best:
        best = evl
        torch.save(net.state_dict(), os.path.join(D, 'gru64_v3.pt'))
print('best', best, flush=True)
