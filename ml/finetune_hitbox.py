"""Phase 4: finetune with hitbox-aware loss.
AABB 48x48 (24 to each side, Chebyshev) horizontally; projectile volume ignored.
  err <= 24 : L = ln((err+1)/25)      # log reward, steep near perfect (-3.2..0)
  err >  24 : L = 4*((err-24)/24)^2   # quadrupling penalty outside (0..inf)
Mean over horizons + small single-step anchor. Binary hit labels fall out free
for the future value head.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
tr = np.load(os.path.join(D, 'train3.npz'), mmap_mode='r')
ev = np.load(os.path.join(D, 'eval3.npz'), mmap_mode='r')
Xtr, Ytr, Wtr, Htr = tr['X'], tr['Y'], tr['W'], tr['H']
Xev, Yev, Wev, Hev = ev['X'], ev['Y'], ev['W'], ev['H']
mtr = (np.abs(Ytr).max(axis=1) <= 40)
mev = (np.abs(Yev).max(axis=1) <= 40)
Xtr, Ytr, Wtr, Htr = Xtr[mtr], Ytr[mtr], Wtr[mtr], Htr[mtr]
Xev, Yev, Wev, Hev = Xev[mev], Yev[mev], Wev[mev], Hev[mev]
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HB = 24.0


class Net(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.gru = nn.GRU(5, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def unroll(net, xb):
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
    return torch.stack(traj, dim=1), net.head(hh[-1])


def hitbox_loss(net, xb, yb, wb, Hb):
    traj, ahead = unroll(net, xb)
    l = 0.0
    for hi, h_ in enumerate(HOR):
        tgt = torch.from_numpy(np.array(Hb[:, hi]))
        m = (~torch.isnan(tgt[:, 0:1])).float()
        err = torch.abs(traj[:, h_ - 1] - tgt).amax(dim=1, keepdim=True)  # Chebyshev = AABB
        inside = (err <= HB).float()
        lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
        lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
        l = l + ((lin + lout) * m).sum() / m.sum().clamp_min(1)
    l = l / len(HOR)
    anchor = (((ahead - yb) ** 2) * wb).mean() * 0.05
    return l + anchor


net = Net()
net.load_state_dict(torch.load(os.path.join(D, 'gru64_v3ii.pt'), map_location='cpu'))
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
B = 128
N = len(Xtr)
best = 1e18
for ep in range(4):
    net.train()
    perm = np.random.permutation(N)
    tot = 0.0
    t0 = time.time()
    for i in range(0, N, B):
        idx = perm[i:i + B]
        xb = torch.from_numpy(np.array(Xtr[idx]))
        yb = torch.from_numpy(np.array(Ytr[idx]))
        wb = torch.from_numpy(np.array(Wtr[idx])).view(-1, 1)
        Hb = Htr[idx]
        opt.zero_grad()
        l = hitbox_loss(net, xb, yb, wb, Hb)
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot += l.item() * len(idx)
    net.eval()
    with torch.no_grad():
        evl = 0.0
        nb = 0
        for i in range(0, len(Xev), 512):
            xb = torch.from_numpy(np.array(Xev[i:i + 512]))
            yb = torch.from_numpy(np.array(Yev[i:i + 512]))
            wb = torch.from_numpy(np.array(Wev[i:i + 512])).view(-1, 1)
            Hb = Hev[i:i + 512]
            evl += hitbox_loss(net, xb, yb, wb, Hb).item() * len(xb)
            nb += len(xb)
        evl /= nb
    print(f'ep{ep} train_hb={tot / N:.4f} eval_hb={evl:.4f} {time.time() - t0:.0f}s', flush=True)
    if evl < best:
        best = evl
        torch.save(net.state_dict(), os.path.join(D, 'gru64_v4.pt'))
print('best', best, flush=True)
