"""Hitbox-loss finetune for ctx models. Usage: finetune_ctx.py [raw|res].
res: net predicts correction; unroll adds heur back (from static ctx feats 32:34).
Saves gru_ctx_<mode>_v5.pt."""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
MODE = sys.argv[1] if len(sys.argv) > 1 else 'raw'
CKPT = os.path.join(D, 'rawctx.pt' if MODE == 'raw' else 'resctx.pt')


def load(tag):
    d = np.load(os.path.join(D, tag + '3.npz'), mmap_mode='r')
    e = np.load(os.path.join(D, tag + '_ctx.npz'), mmap_mode='r')
    Xb = np.array(d['X'])
    Es = np.repeat(np.array(e['E'])[:, None, :], Xb.shape[1], axis=1)
    X = np.concatenate([Xb, Es], axis=2)
    return X, np.array(d['Y']), np.array(d['W']), np.array(d['H'])


Xtr, Ytr, Wtr, Htr = load('train')
Xev, Yev, Wev, Hev = load('eval')
mtr = (np.abs(Ytr).max(axis=1) <= 40)
mev = (np.abs(Yev).max(axis=1) <= 40)
Xtr, Ytr, Wtr, Htr = Xtr[mtr], Ytr[mtr], Wtr[mtr], Htr[mtr]
Xev, Yev, Wev, Hev = Xev[mev], Yev[mev], Wev[mev], Hev[mev]
print(MODE, 'train', Xtr.shape, 'eval', Xev.shape, flush=True)
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HB = 24.0


class Net(nn.Module):
    def __init__(self, h=80):
        super().__init__()
        self.gru = nn.GRU(37, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def unroll(net, xb):
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * 240.0
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:]  # 32 ctx feats, frozen during rollout
    heur = xb[:, -1, 32:34] * 240.0 if MODE == 'res' else 0.0
    hh = h
    traj = []
    for _ in range(66):
        a = net.head(hh[-1]) * 240.0 + heur
        v = v + a * DT
        p = p + v * DT
        sp = (v ** 2).sum(-1, keepdim=True).sqrt() / 240.0
        dyn = torch.cat([v / 240.0, sp, gr, torch.zeros_like(sp)], dim=-1).unsqueeze(1)
        xin = torch.cat([dyn, static], dim=-1)
        _, hh = net.gru(xin, hh)
        traj.append(p)
    return torch.stack(traj, dim=1), net.head(hh[-1]) * 240.0 + heur


def hitbox_loss(net, xb, yb, wb, Hb):
    traj, ahead = unroll(net, xb)
    tgt_acc = yb * 240.0 + (xb[:, -1, 32:34] * 240.0 if MODE == 'res' else 0.0)
    l = 0.0
    for hi, h_ in enumerate(HOR):
        tgt = torch.from_numpy(np.array(Hb[:, hi]))
        m = (~torch.isnan(tgt[:, 0:1])).float()
        err = torch.abs(traj[:, h_ - 1] - tgt).amax(dim=1, keepdim=True)
        inside = (err <= HB).float()
        lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
        lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
        l = l + ((lin + lout) * m).sum() / m.sum().clamp_min(1)
    l = l / len(HOR)
    anchor = ((((ahead - tgt_acc) / 240.0) ** 2) * wb).mean() * 0.05
    return l + anchor


net = Net()
net.load_state_dict(torch.load(CKPT, map_location='cpu'))
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
B = 128
N = len(Xtr)
best = 1e18
OUT = os.path.join(D, f'gru_ctx_{MODE}_v5.pt')
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
    print(f'{MODE} ep{ep} train_hb={tot / N:.4f} eval_hb={evl:.4f} {time.time() - t0:.0f}s', flush=True)
    if evl < best:
        best = evl
        torch.save(net.state_dict(), OUT)
print(MODE, 'best', best, flush=True)
