"""Value head on denseC champion (frozen policy, friction-order unroll).
Features per sample: warmup hidden (96) + rollout endpoint stats
(endpoint dist, endpoint speed, mean|accel|, grounded, horizon_T).
Target: binary hit (inside 24hu AABB, Chebyshev) per horizon [7,13,20,33,66].
Train: BCE, 4000 soldier train rows. Eval: 2000 rows, threshold sweep per horizon
(coverage vs gated hit-rate) + calibration bins.
Output: value_denseC.pt (MLP 101->32->5). Release gate consumes P(hit)."""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0
MAXV = 240.0


class Pol(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(14, 96, batch_first=True)
        self.head = nn.Linear(96, 2)


class Val(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(96 + 5, 32), nn.ReLU(), nn.Linear(32, 5))

    def forward(self, x):
        return self.mlp(x)


def apply_friction(v, gr):
    sp = (v ** 2).sum(-1, keepdim=True).sqrt()
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


pol = Pol()
pol.load_state_dict(torch.load(os.path.join(D, 'gru_soldier_denseC.pt'), map_location='cpu'))
pol.eval()


def feats_hits(Xs, Hs):
    """Rollout with FRICTION (matches denseC training). Returns feats (N,101), hits (N,5)."""
    Fall, Yall = [], []
    with torch.no_grad():
        for b in range(0, len(Xs), 128):
            xb = torch.from_numpy(Xs[b:b + 128])
            Hb = Hs[b:b + 128]
            _, h = pol.gru(xb)
            h96 = h[-1]
            v = xb[:, -1, :2].clone() * MAXV
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            hh = h
            hs, aa = [], []
            for _ in range(66):
                a = pol.head(hh[-1]) * MAXV
                aa.append(a.clone())
                v = apply_friction(v, gr) + a * DT
                p = p + v * DT
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / MAXV
                xin = torch.cat([(v / MAXV).unsqueeze(1), sp.unsqueeze(1),
                                 gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
                _, hh = pol.gru(torch.cat([xin, xb[:, -1:, 5:]], dim=-1), hh)
                hs.append(p.clone())
            traj = torch.stack(hs, dim=1).numpy()
            acc = torch.stack(aa, dim=1)
            ep_dist = np.linalg.norm(traj[:, -1], axis=1, keepdims=True)
            ep_sp = np.linalg.norm((xb[:, -1, :2].numpy() * MAXV), axis=1, keepdims=True)
            meana = acc.abs().amax(dim=1).numpy().mean(axis=1, keepdims=True)
            stat = np.concatenate([ep_dist / 1000.0, ep_sp / 1000.0, meana / 2000.0,
                                   xb[:, -1, 3:4].numpy(),
                                   np.full((len(xb), 1), 66 * DT, dtype=np.float32)], axis=1)
            Fall.append(np.concatenate([h96.numpy(), stat], axis=1))
            Yb = np.zeros((len(xb), len(HOR)), dtype=np.float32)
            for hi, h_ in enumerate([6, 12, 19, 32, 65]):  # dense cols for HOR ticks
                tgt = Hb[:, h_]
                mm = ~np.isnan(tgt[:, 0])
                err = np.abs(traj[:, h_ - 1] - tgt).max(axis=1)
                Yb[mm, hi] = (err[mm] < HITR).astype(np.float32)
            Yall.append(Yb)
    return np.concatenate(Fall), np.concatenate(Yall)


def load_xy(tag, n, seed):
    d = np.load(os.path.join(D, 'trainB.npz' if tag == 'train' else 'evalB.npz'), mmap_mode='r')
    k = np.load(os.path.join(D, tag + 'Bk.npz'))['K']
    dn = np.load(os.path.join(D, tag + 'B_dense66.npz'))
    X, H = np.array(d['X']), dn['Hd']
    sol = k[:, 3] == 3
    Xs, Hs = X[sol], H
    assert len(Xs) == len(Hs), (len(Xs), len(Hs))
    idx = np.random.RandomState(seed).choice(len(Xs), min(n, len(Xs)), replace=False)
    return Xs[idx], Hs[idx]


print('building features (train)...', flush=True)
Xtr, Htr = load_xy('train', 4000, 0)
Ftr, Ytr = feats_hits(Xtr, Htr)
print('building features (eval)...', flush=True)
Xev, Hev = load_xy('eval', 2000, 1)
Fev, Yev = feats_hits(Xev, Hev)
print('base hit rates:', Ytr.mean(axis=0).round(3), flush=True)

val = Val()
opt = torch.optim.Adam(val.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()
Fb = torch.from_numpy(Ftr)
Yb = torch.from_numpy(Ytr)
for ep in range(40):
    opt.zero_grad()
    l = bce(val(Fb), Yb)
    l.backward()
    opt.step()
    if ep % 10 == 0:
        print(f'ep{ep} bce={l.item():.4f}', flush=True)
torch.save(val.state_dict(), os.path.join(D, 'value_denseC.pt'))

with torch.no_grad():
    P = torch.sigmoid(val(torch.from_numpy(Fev))).numpy()
print('threshold sweep per horizon (coverage% | gated-hit%):', flush=True)
names = ['100', '200', '300', '500', '1000']
for hi, nm in enumerate(names):
    line = []
    for thr in [0.0, 0.3, 0.5, 0.7, 0.9]:
        m = P[:, hi] >= thr
        cov = m.mean()
        hr = Yev[m, hi].mean() if m.sum() else float('nan')
        line.append(f'{thr}:{cov * 100:.0f}%/{hr * 100:.0f}%')
    print(f'  h={nm}: base={Yev[:, hi].mean() * 100:.1f}% ' + ' '.join(line), flush=True)
print('calibration @300ms:', flush=True)
for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
    m = (P[:, 2] >= lo) & (P[:, 2] < hi)
    if m.sum():
        print(f'  bin[{lo},{hi}): n={m.sum()} pred={P[m, 2].mean():.2f} actual={Yev[m, 2].mean():.2f}', flush=True)
