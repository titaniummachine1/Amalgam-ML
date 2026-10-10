"""Value head on the threatsimB CHAMPION (frozen policy, threat-sim unroll).
Same protocol as value_denseC.py: per-sample feats = warmup hidden (96) +
rollout endpoint stats (5); targets = binary hit @24hu per horizon [7,13,20,33,66].
Train BCE on 4000 soldier train rows; eval 2000 rows + threshold sweep +
calibration. Output: value_simB.pt (MLP 101->32->5). Powers the ML-hitchance
gate (m_flConfidence 0..1 x 100 vs HitChance slider).
Memory-safe: memmap + scattered subset only (never full np.array of 12GB X).
Usage: python ml/value_simB.py"""
import os

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
HCOLS = [6, 12, 19, 32, 65]
DT = 1 / 66.0
HITR = 24.0
MAXV = 240.0


class Pol(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(20, 96, batch_first=True)
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
pol.load_state_dict(torch.load(os.path.join(D, 'gru_threatsimB.pt'), map_location='cpu'))
pol.eval()


def feats_hits(Xs, Ss, Hs):
    Fall, Yall = [], []
    with torch.no_grad():
        for b in range(0, len(Xs), 128):
            xb = torch.from_numpy(np.array(Xs[b:b + 128]))
            sb = torch.from_numpy(np.array(Ss[b:b + 128]))
            Hb = Hs[b:b + 128]
            _, h = pol.gru(xb)
            h96 = h[-1]
            v = xb[:, -1, :2].clone() * MAXV
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            static = xb[:, -1:, 5:14]
            thr_frozen = xb[:, -1:, 14:20]
            q = sb[:, 0:2].clone()
            vth = sb[:, 2:4].clone()
            present = sb[:, 4:5]
            nspd = torch.clamp((vth ** 2).sum(-1, keepdim=True).sqrt() / 3000.0, 0.0, 1.0)
            cnt = thr_frozen[:, 0, 3:4]
            mem = thr_frozen[:, 0, 4:5]
            hh = h
            hs, aa = [], []
            for _ in range(66):
                a = pol.head(hh[-1]) * MAXV
                aa.append(a.clone())
                v = apply_friction(v, gr) + a * DT
                p = p + v * DT
                q = q + vth * DT
                rel = q - p
                dist = rel.norm(dim=1, keepdim=True).clamp_min(1e-6)
                ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
                ang = torch.atan2(rel[:, 1:2], rel[:, 0:1])
                infov = (ang.abs() < (np.pi / 4)).float()
                appr = -((vth - v) * rel).sum(-1, keepdim=True) / dist
                tti = torch.where(appr > 50.0, torch.clamp(dist / appr / 5.0, 0.0, 1.0),
                                  torch.ones_like(dist))
                live = torch.cat([ndist, nspd, infov, cnt, mem, tti], dim=-1).unsqueeze(1)
                thr = present.unsqueeze(1) * live + (1.0 - present.unsqueeze(1)) * thr_frozen
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / MAXV
                dyn = torch.cat([(v / MAXV).unsqueeze(1), sp.unsqueeze(1),
                                 gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1),
                                 thr], dim=-1)
                _, hh = pol.gru(torch.cat([dyn, static], dim=-1), hh)
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
            for hi, h_ in enumerate(HCOLS):
                tgt = Hb[:, h_]
                mm = ~np.isnan(tgt[:, 0])
                err = np.abs(traj[:, h_ - 1] - tgt).max(axis=1)
                Yb[mm, hi] = (err[mm] < HITR).astype(np.float32)
            Yall.append(Yb)
    return np.concatenate(Fall), np.concatenate(Yall)


def load_xy(tag, n, seed):
    d = np.load(os.path.join(D, 'trainB.npz' if tag == 'train' else 'evalB.npz'), mmap_mode='r')
    k = np.load(os.path.join(D, tag + 'Bk.npz'))['K']
    tw = np.load(os.path.join(D, tag + 'B_threatwin.npz'), mmap_mode='r')
    dn = np.load(os.path.join(D, tag + 'B_dense66.npz'), mmap_mode='r')
    sol = np.where(k[:, 3] == 3)[0]
    idx = np.random.RandomState(seed).choice(len(sol), min(n, len(sol)), replace=False)
    rows = sol[idx]
    X = np.concatenate([np.array(d['X'][rows]), np.array(tw['TW'][idx])], axis=2)
    S = np.array(tw['TS'][idx])[:, -1, :]
    H = np.array(dn['Hd'][idx])
    return X, S, H


print('building features (train)...', flush=True)
Xtr, Str, Htr = load_xy('train', 4000, 0)
Ftr, Ytr = feats_hits(Xtr, Str, Htr)
print('building features (eval)...', flush=True)
Xev, Sev, Hev = load_xy('eval', 2000, 1)
Fev, Yev = feats_hits(Xev, Sev, Hev)
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
        print('ep%d bce=%.4f' % (ep, l.item()), flush=True)
torch.save(val.state_dict(), os.path.join(D, 'value_simB.pt'))

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
        hrs = '%d' % (hr * 100) if hr == hr else 'nan'
        line.append('%.1f:%d%%/%s%%' % (thr, cov * 100, hrs))
    print('  h=%s: base=%.1f%% %s' % (nm, Yev[:, hi].mean() * 100, ' '.join(line)), flush=True)
print('calibration @300ms:', flush=True)
for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
    m = (P[:, 2] >= lo) & (P[:, 2] < hi)
    if m.sum():
        print('  bin[%.1f,%.1f): n=%d pred=%.2f actual=%.2f' % (
            lo, hi, m.sum(), P[m, 2].mean(), Yev[m, 2].mean()), flush=True)
