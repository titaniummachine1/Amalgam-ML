"""Value head on the DEPLOYED denseE policy (frozen, 21-dim E unroll, weapon
frozen @T). Same protocol as value_denseC: feats = warmup hidden (96) + rollout
endpoint stats (5); targets = binary hit @24hu per horizon [7,13,20,33,66].
Data: trainE/evalE (X18+WLD->21) + Cmask kept rows; Hd targets built on-demand
from dataset_all.db for the 4000/2000 subset only (E Hd caches are stale vs the
rebuilt B). Train BCE 40 epochs. OUT: value_denseE.pt. Powers ML-hitchance.
Usage: python ml/value_denseE.py"""
import os
import sqlite3
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DS = D
HOR = [7, 13, 20, 33, 66]
HCOLS = [6, 12, 19, 32, 65]
DT = 1 / 66.0
HITR = 24.0
MAXV1 = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)
TELEPORT = 300.0
con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))


class Pol(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(21, 96, batch_first=True)
        self.head = nn.Linear(96, 2)


class Val(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(96 + 5, 32), nn.ReLU(), nn.Linear(32, 5))

    def forward(self, x):
        return self.mlp(x)


def apply_friction(v, gr):
    sp = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-12)
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


pol = Pol()
pol.load_state_dict(torch.load(os.path.join(D, 'gru_soldier_denseE.pt'), map_location='cpu'))
pol.eval()


def unroll(net, xb):
    mv = (xb[:, -1:, 5:14] * torch.from_numpy(MAXV1)).sum(-1, keepdim=True)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mv.squeeze(1)
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:14]
    wstat = xb[:, -1:, 14:21]
    hh = h
    hs, aa = [], []
    for _ in range(66):
        a = net.head(hh[-1]) * mv.squeeze(1)
        aa.append(a.clone())
        v = apply_friction(v, gr) + a * DT
        p = p + v * DT
        sp = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-12) / mv.squeeze(1)
        dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static, wstat], dim=-1), hh)
        hs.append(p.clone())
    return torch.stack(hs, dim=1), torch.stack(aa, dim=1), h[-1]


def build_targets(keys):
    """keys: list of (did,ent,t). Returns Hd (M,66,2). Ego frame = dense66 rule."""
    groups = defaultdict(list)
    for i, (did, ent, t) in enumerate(keys):
        groups[(int(did), int(ent))].append((i, int(t)))
    Hd = np.full((len(keys), 66, 2), np.nan, dtype=np.float32)
    for (did, ent), idxs in groups.items():
        rows = con.execute(
            'SELECT tick,ox,oy,eyey FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
            (did, ent)).fetchall()
        if not rows:
            continue
        tk = np.array([r[0] for r in rows])
        ox = np.array([r[1] for r in rows])
        oy = np.array([r[2] for r in rows])
        ey = np.array([r[3] for r in rows])
        pos = {t: j for j, t in enumerate(tk)}
        for i, t in idxs:
            jT = pos.get(t)
            if jT is None or jT < 1 or jT + 66 > len(tk):
                continue
            if np.any(tk[jT + 1:jT + 67] != t + 1 + np.arange(66)):
                continue
            if np.hypot(np.diff(ox[jT:jT + 67]), np.diff(oy[jT:jT + 67])).max() > TELEPORT:
                continue
            vlt = np.array([ox[jT] - ox[jT - 1], oy[jT] - oy[jT - 1]]) / DT
            yaw = ey[jT]
            if not np.isfinite(yaw):
                yaw = 0.0
            if np.linalg.norm(vlt) > 50:
                yaw = np.rad2deg(np.arctan2(vlt[1], vlt[0]))
            a = np.deg2rad(yaw)
            c, sn = np.cos(-a), np.sin(-a)
            R = np.array([[c, -sn], [sn, c]])
            for hi, h in enumerate(range(1, 67)):
                d = R @ np.array([ox[jT + h] - ox[jT], oy[jT + h] - oy[jT]])
                Hd[i, hi, 0] = d[0]
                Hd[i, hi, 1] = d[1]
    return Hd


def load_xy(tag, n, seed):
    d = np.load(os.path.join(D, tag + 'E.npz'), mmap_mode='r')
    kk = np.load(os.path.join(D, tag + 'Ek.npz'))['K']
    Wm = np.load(os.path.join(D, tag + 'E_wld.npz'), mmap_mode='r')['WLD']
    mk = np.load(os.path.join(D, tag + 'Cmask.npz'))['keep']
    sol = kk[:, 3] == 3
    good = np.where(sol & mk)[0]
    idx = np.random.RandomState(seed).choice(good, min(n, len(good)), replace=False)
    X = np.concatenate([np.array(d['X'][idx]),
                        np.array(Wm[idx]).astype(np.float32) / 16.0], axis=2)
    keys = [(kk[j][0], kk[j][1], kk[j][2]) for j in idx]
    H = build_targets(keys)
    m = ~np.isnan(H[:, 65, 0])
    print(tag, 'kept %d/%d with full 66-future' % (m.sum(), len(idx)), flush=True)
    return X[m], H[m]


def feats_hits(Xs, Hs):
    Fall, Yall = [], []
    with torch.no_grad():
        for b in range(0, len(Xs), 128):
            xb = torch.from_numpy(np.array(Xs[b:b + 128]))
            Hb = Hs[b:b + 128]
            traj, acc, h96 = unroll(pol, xb)
            ep_dist = np.linalg.norm(traj[:, -1].numpy(), axis=1, keepdims=True)
            ep_sp = np.linalg.norm((xb[:, -1, :2].numpy() * 240.0), axis=1, keepdims=True)
            meana = acc.abs().amax(dim=1).numpy().mean(axis=1, keepdims=True)
            stat = np.concatenate([ep_dist / 1000.0, ep_sp / 1000.0, meana / 2000.0,
                                   xb[:, -1, 3:4].numpy(),
                                   np.full((len(xb), 1), 66 * DT, dtype=np.float32)], axis=1)
            Fall.append(np.concatenate([h96.numpy(), stat], axis=1))
            Yb = np.zeros((len(xb), len(HOR)), dtype=np.float32)
            for hi, h_ in enumerate(HCOLS):
                tgt = Hb[:, h_]
                mm = ~np.isnan(tgt[:, 0])
                err = np.abs(traj[:, h_ - 1].numpy() - tgt).max(axis=1)
                Yb[mm, hi] = (err[mm] < HITR).astype(np.float32)
            Yall.append(Yb)
    return np.concatenate(Fall), np.concatenate(Yall)


print('loading (train)...', flush=True)
Xtr, Htr = load_xy('train', 4000, 0)
print('rolling (train)...', flush=True)
Ftr, Ytr = feats_hits(Xtr, Htr)
print('loading (eval)...', flush=True)
Xev, Hev = load_xy('eval', 2000, 1)
print('rolling (eval)...', flush=True)
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
        print('ep%d bce=%.4f' % (ep, l.item()), flush=True)
torch.save(val.state_dict(), os.path.join(D, 'value_denseE.pt'))

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
        hrs = ('%d' % (hr * 100)) if hr == hr else 'nan'
        line.append('%.1f:%d%%/%s%%' % (thr, cov * 100, hrs))
    print('  h=%s: base=%.1f%% %s' % (nm, Yev[:, hi].mean() * 100, ' '.join(line)), flush=True)
print('calibration @300ms:', flush=True)
for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
    m = (P[:, 2] >= lo) & (P[:, 2] < hi)
    if m.sum():
        print('  bin[%.1f,%.1f): n=%d pred=%.2f actual=%.2f' % (
            lo, hi, m.sum(), P[m, 2].mean(), Yev[m, 2].mean()), flush=True)
