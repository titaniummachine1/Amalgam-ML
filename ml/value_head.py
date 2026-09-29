"""Value head prototype (v4 soldier policy, frozen).
Features per sample: warmup hidden (64) + rollout endpoint stats (dist, speed,
mean|accel|, grounded, horizon). Target: binary hit (inside 24hu AABB) per horizon.
Output: P(hit) per horizon -> release gate: shoot only if conf >= threshold.
Reports gated hit-rate/coverage curve. -> value_v4.pt"""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0


class Pol(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(5, 64, batch_first=True)
        self.head = nn.Linear(64, 2)


class Val(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(64 + 5, 32), nn.ReLU(), nn.Linear(32, 5))

    def forward(self, x):
        return self.mlp(x)


pol = Pol()
pol.load_state_dict(torch.load(os.path.join(D, 'gru64_v4.pt'), map_location='cpu'))
pol.eval()


# recompute trajectories for labels (same loop, keep traj)
def feats_hits(Xs, Hs):
    d = np.load(os.path.join(D, tag + '3.npz'), mmap_mode='r')
    X, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
    m = (np.abs(Y).max(axis=1) <= 40)
    X, H = X[m], H[m]
    idx = np.random.RandomState(seed).choice(len(X), min(n, len(X)), replace=False)
    return X[idx], H[idx]


print('building features...', flush=True)
Xtr, Htr = load_xy('train', 4000, 0)
Xev, Hev = load_xy('eval', 2000, 1)
Ftr = features(Xtr)
Fev = features(Xev)


def hits(H):
    Y = np.zeros((len(H), len(HOR)), dtype=np.float32)
    for hi, h_ in enumerate(HOR):
        tgt = H[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        # placeholder: need traj; recompute compactly via coast? No -> use stored below.
    return Y


# recompute trajectories for labels (same loop, keep traj)
def feats_hits(Xs, Hs):
    Fall, Yall = [], []
    with torch.no_grad():
        for b in range(0, len(Xs), 128):
            xb = torch.from_numpy(Xs[b:b + 128])
            Hb = Hs[b:b + 128]
            _, h = pol.gru(xb)
            h64 = h[-1]
            v = xb[:, -1, :2].clone() * 240.0
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            hh = h
            hs = []
            aa = []
            for _ in range(66):
                a = pol.head(hh[-1]) * 240.0
                aa.append(a.clone())
                v = v + a * DT
                p = p + v * DT
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / 240.0
                xin = torch.cat([v / 240.0, sp, gr, torch.zeros_like(sp)], dim=-1).unsqueeze(1)
                _, hh = pol.gru(xin, hh)
                hs.append(p.clone())
            traj = torch.stack(hs, dim=1).numpy()
            acc = torch.stack(aa, dim=1)
            ep_dist = np.linalg.norm(traj[:, -1], axis=1, keepdims=True)
            ep_sp = np.linalg.norm((xb[:, -1, :2].numpy() * 240.0), axis=1, keepdims=True)
            meana = acc.abs().amax(dim=1).numpy().mean(axis=1, keepdims=True)
            stat = np.concatenate([ep_dist / 1000.0, ep_sp / 1000.0, meana / 2000.0,
                                   xb[:, -1, 3:4].numpy(),
                                   np.full((len(xb), 1), 66 * DT, dtype=np.float32)], axis=1)
            Fall.append(np.concatenate([h64.numpy(), stat], axis=1))
            Yb = np.zeros((len(xb), len(HOR)), dtype=np.float32)
            for hi, h_ in enumerate(HOR):
                tgt = Hb[:, hi]
                mm = ~np.isnan(tgt[:, 0])
                err = np.abs(traj[:, h_ - 1] - tgt).max(axis=1)
                Yb[mm, hi] = (err[mm] < HITR).astype(np.float32)
            Yall.append(Yb)
    return np.concatenate(Fall), np.concatenate(Yall)


print('rollouts for labels...', flush=True)
Ftr, Ytr = feats_hits(Xtr, Htr)
Fev, Yev = feats_hits(Xev, Hev)
print('base rates:', Ytr.mean(axis=0).round(3), flush=True)

val = Val()
opt = torch.optim.Adam(val.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()
Fb = torch.from_numpy(Ftr)
Yb = torch.from_numpy(Ytr)
for ep in range(30):
    opt.zero_grad()
    l = bce(val(Fb), Yb)
    l.backward()
    opt.step()
    if ep % 10 == 0:
        print(f'ep{ep} bce={l.item():.4f}', flush=True)
torch.save(val.state_dict(), os.path.join(D, 'value_v4.pt'))

with torch.no_grad():
    P = torch.sigmoid(val(torch.from_numpy(Fev))).numpy()
print('threshold sweep @300ms horizon (hi=2):', flush=True)
for thr in [0.0, 0.3, 0.5, 0.7, 0.9]:
    m = P[:, 2] >= thr
    cov = m.mean()
    hr = Yev[m, 2].mean() if m.sum() else float('nan')
    print(f'  thr={thr}: coverage={cov * 100:.1f}% gated-hitrate={hr * 100:.1f}%', flush=True)
# calibration: mean predicted vs actual per bin
for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
    m = (P[:, 2] >= lo) & (P[:, 2] < hi)
    if m.sum():
        print(f'  bin[{lo},{hi}): n={m.sum()} pred={P[m, 2].mean():.2f} actual={Yev[m, 2].mean():.2f}', flush=True)
