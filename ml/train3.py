"""Train GRU-64 policy on dataset v2. Weighted accel MSE + multi-horizon
autoregressive rollout position error vs coast / yaw-rate baselines."""
import os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
tr = np.load(os.path.join(D, 'train3.npz'), mmap_mode='r')
ev = np.load(os.path.join(D, 'eval3.npz'), mmap_mode='r')
Xtr, Ytr, Wtr = tr['X'], tr['Y'], tr['W']
Xev, Yev, Wev, Hev = ev['X'], ev['Y'], ev['W'], ev['H']
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)

# drop teleport-residue labels
mtr = (np.abs(Ytr).max(axis=1) <= 40)
mev = (np.abs(Yev).max(axis=1) <= 40)
Xtr, Ytr, Wtr = Xtr[mtr], Ytr[mtr], Wtr[mtr]
Xev, Yev, Wev, Hev = Xev[mev], Yev[mev], Wev[mev], Hev[mev]
print('after |Y|<=40 filter:', Xtr.shape, Xev.shape, flush=True)

def wmse(P, Y, W):
    return float(((P - Y) ** 2).mean(axis=1).__mul__(W).mean() / W.mean())

def baseline_accel(X, Y, W, tag):
    coast = wmse(np.zeros_like(Y), Y, W)
    v = X[:, -8:, :2]
    hd = np.arctan2(v[:, :, 1], v[:, :, 0])
    dh = (np.diff(hd, axis=1) + np.pi) % (2 * np.pi) - np.pi
    w = dh.mean(axis=1)
    c, s = np.cos(w), np.sin(w)
    v0 = X[:, -1, :2]
    ap = (np.stack([c * v0[:, 0] - s * v0[:, 1] - v0[:, 0],
                    s * v0[:, 0] + c * v0[:, 1] - v0[:, 1]], axis=1)) * 66.0
    print(f'{tag} coast_wMSE={coast:.4f} yawrate_wMSE={wmse(ap, Y, W):.4f}', flush=True)

baseline_accel(Xtr, Ytr, Wtr, 'train')
baseline_accel(Xev, Yev, Wev, 'eval')

HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0

def rollout_position_errors(net, X, H, n=2000, tag='eval'):
    """Autoregressive rollout in ego frame, frozen grounded/yaw. Returns mean err per horizon (hu)."""
    net.eval()
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    Xs, Hs = X[idx], H[idx]
    errs = np.zeros((len(HOR),))
    counts = np.zeros((len(HOR),))
    with torch.no_grad():
        for b in range(0, len(Xs), 128):
            xb = torch.from_numpy(Xs[b:b + 128])
            out, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * 240.0  # ego hu/s... note: features normalized; de-norm
            p = torch.zeros_like(v)
            # per-step predictions
            hs = []
            hh = h
            gr_frozen = xb[:, -1, 3:4]
            for _ in range(66):
                a = net.head(hh[-1]) * 240.0  # ego hu/s^2
                v = v + a * DT
                p = p + v * DT
                # feed back: update vel/speed feats, freeze grounded/dyaw
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / 240.0
                xin = torch.cat([v / 240.0, sp, gr_frozen,
                                 torch.zeros_like(sp)], dim=-1).unsqueeze(1)
                _, hh = net.gru(xin, hh)
                hs.append(p.clone())
            traj = torch.stack(hs, dim=1).numpy()  # (B,66,2) ego hu
            for hi, hh_ in enumerate(HOR):
                tgt = Hs[b:b + 128, hi] * 1.0  # H stored in hu? (R @ delta-hu) yes hu
                m = ~np.isnan(tgt[:, 0])
                if m.sum():
                    errs[hi] += np.abs(traj[m, hh_ - 1] - tgt[m]).mean() * m.sum()
                    counts[hi] += m.sum()
    errs /= np.maximum(counts, 1)
    print(tag, 'rollout pos err hu @100/200/300/500/1000ms:',
          ' '.join(f'{e:.1f}' for e in errs), flush=True)
    return errs

def baseline_rollout(X, H, n=2000, tag='eval'):
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    Xs, Hs = X[idx] * 1.0, H[idx]
    v0 = Xs[:, -1, :2] * 240.0
    errs = np.zeros(len(HOR))
    # yaw-rate from last 8 headings
    v = Xs[:, -8:, :2]
    hd = np.arctan2(v[:, :, 1], v[:, :, 0])
    dh = (np.diff(hd, axis=1) + np.pi) % (2 * np.pi) - np.pi
    w = dh.mean(axis=1)
    for hi, h in enumerate(HOR):
        # coast
        pc = v0 * (h * DT)
        # yawrate integrate
        t = np.arange(1, h + 1) * DT
        ang = w[:, None] * t[None, :]
        sp = np.linalg.norm(v0, axis=1, keepdims=True)
        h0 = np.arctan2(v0[:, 1], v0[:, 0])
        px = (sp * np.cos(h0[:, None] + ang)).sum(axis=1) * DT
        py = (sp * np.sin(h0[:, None] + ang)).sum(axis=1) * DT
        py_ = np.stack([px, py], axis=1)
        m = ~np.isnan(Hs[:, hi, 0])
        ec = np.abs(pc[m] - Hs[m, hi]).mean()
        ey = np.abs(py_[m] - Hs[m, hi]).mean()
        print(f'{tag} h={h}: coast={ec:.1f}hu yawrate={ey:.1f}hu', flush=True)

print('--- baselines ---', flush=True)
baseline_rollout(Xev, Hev)

class Net(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.gru = nn.GRU(5, h, batch_first=True)
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
for ep in range(10):
    net.train()
    perm = np.random.permutation(N)
    tot = 0.0; t0 = time.time()
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
    # eval weighted MSE on subset for speed
    net.eval()
    with torch.no_grad():
        evl = 0.0; nb = 0
        for i in range(0, len(Xev), 2048):
            xb = torch.from_numpy(np.array(Xev[i:i + 2048]))
            yb = torch.from_numpy(np.array(Yev[i:i + 2048]))
            wb = torch.from_numpy(np.array(Wev[i:i + 2048])).view(-1, 1)
            evl += ((lossf(net(xb), yb) * wb).mean(axis=1).mean() * len(xb))
            nb += len(xb)
        evl = float(evl / nb)
    print(f'ep{ep} train_wMSE={tot / N:.4f} eval_wMSE={evl:.4f} {time.time() - t0:.0f}s', flush=True)
    if evl < best:
        best = evl
        torch.save(net.state_dict(), os.path.join(D, 'gru64_v2.pt'))
print('best', best, flush=True)
print('--- model rollout ---', flush=True)
net.load_state_dict(torch.load(os.path.join(D, 'gru64_v2.pt')))
rollout_position_errors(net, np.array(Xev), np.array(Hev))
