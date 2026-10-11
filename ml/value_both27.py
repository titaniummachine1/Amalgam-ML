"""Value head on the both27 CHAMPION (frozen, 27-dim threat+weapon unroll).
Same protocol as value_simB: feats = warmup hidden (96) + rollout endpoint
stats (5); targets = binary hit @24hu per horizon [7,13,20,33,66].
Data: repack X20 (1.34M) + W7 slices for a 4000/2000 subset (soldier order,
no key mapping needed); Hd from HD66 repack. Train BCE 40 epochs.
OUT: value_both27.pt. Usage: python ml/value_both27.py"""
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
MAXV1 = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)
_LN_EPS = 1e-12


def slen(v, dim=-1):
    return torch.sqrt((v ** 2).sum(dim=dim, keepdim=True) + _LN_EPS)


class Pol(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(27, 96, batch_first=True)
        self.head = nn.Linear(96, 2)


class Val(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(96 + 5, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x):
        return self.mlp(x)


def apply_friction(v, gr):
    sp = slen(v)
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


pol = Pol()
pol.load_state_dict(torch.load(os.path.join(D, 'gru_both27_best.pt'), map_location='cpu'))
pol.eval()


def unroll(net, xb, s0):
    mv = (xb[:, -1:, 5:14] * torch.from_numpy(MAXV1)).sum(-1, keepdim=True)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mv.squeeze(1)
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:14]
    wstat = xb[:, -1:, 20:27]
    q = s0[:, 0:2].clone()
    vth = s0[:, 2:4].clone()
    present = s0[:, 4:5]
    thr_frozen = xb[:, -1:, 14:20]
    nspd = torch.clamp(slen(vth) / 3000.0, 0.0, 1.0)
    cnt = thr_frozen[:, 0, 3:4]
    mem = thr_frozen[:, 0, 4:5]
    hh = h
    hs, aa = [], []
    for _ in range(66):
        a = net.head(hh[-1]) * mv.squeeze(1)
        aa.append(a.clone())
        v = apply_friction(v, gr) + a * DT
        p = p + v * DT
        q = q + vth * DT
        rel = q - p
        dist = slen(rel, dim=1).clamp_min(1e-6)
        ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
        ang = torch.atan2(rel[:, 1:2], rel[:, 0:1])
        infov = (ang.abs() < (np.pi / 4)).float()
        appr = -((vth - v) * rel).sum(-1, keepdim=True) / dist
        tti = torch.where(appr > 50.0, torch.clamp(dist / appr.clamp_min(1e-9) / 5.0, 0.0, 1.0),
                          torch.ones_like(dist))
        live = torch.cat([ndist, nspd, infov, cnt, mem, tti], dim=-1).unsqueeze(1)
        thr = present.unsqueeze(1) * live + (1.0 - present.unsqueeze(1)) * thr_frozen
        sp = slen(v) / mv.squeeze(1)
        dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1),
                         thr], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static, wstat], dim=-1), hh)
        hs.append(p.clone())
    traj = torch.stack(hs, dim=1)
    return traj, torch.stack(aa, dim=1), h[-1]


def load_xy(tag, n, seed):
    P = os.path.join(D, tag + 'B_X20sol')
    nn_ = int(os.path.getsize(P + '.raw') // (66 * 20 * 4))
    X = np.memmap(P + '.raw', dtype=np.float32, mode='r', shape=(nn_, 66, 20))
    W7 = np.array(np.load(os.path.join(D, tag + 'B_w7.npz'))['W7'])
    H = np.memmap(P.replace('X20sol', 'HD66') + '.raw', dtype=np.float32, mode='r',
                  shape=(nn_, 66, 2))
    tw = np.load(os.path.join(D, tag + 'B_threatwin.npz'))
    S0 = np.array(tw['TS'])[:, -1, :]
    tw.close()
    idx = np.random.RandomState(seed).choice(nn_, min(n, nn_), replace=False)
    Xb = np.concatenate([np.array(X[idx]), W7[idx]], axis=2)
    return Xb, S0[idx], np.array(H[idx])


def feats_hits(Xs, Ss, Hs):
    Fall, Yall = [], []
    with torch.no_grad():
        for b in range(0, len(Xs), 128):
            xb = torch.from_numpy(np.array(Xs[b:b + 128]))
            sb = torch.from_numpy(np.array(Ss[b:b + 128]))
            Hb = Hs[b:b + 128]
            traj, acc, h96 = unroll(pol, xb, sb)
            ep_dist = np.linalg.norm(traj.numpy()[:, -1], axis=1, keepdims=True)
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
                err = np.abs(traj.numpy()[:, h_ - 1] - tgt).max(axis=1)
                Yb[mm, hi] = (err[mm] < HITR).astype(np.float32)
            Yall.append(Yb)
    return np.concatenate(Fall), np.concatenate(Yall)


print('loading (train)...', flush=True)
Xtr, Str, Htr = load_xy('train', 8000, 0)
print('rolling (train)...', flush=True)
Ftr, Ytr = feats_hits(Xtr, Str, Htr)
print('loading (eval)...', flush=True)
Xev, Sev, Hev = load_xy('eval', 4000, 1)
print('rolling (eval)...', flush=True)
Fev, Yev = feats_hits(Xev, Sev, Hev)
print('base hit rates:', Ytr.mean(axis=0).round(3), flush=True)

val = Val()
opt = torch.optim.Adam(val.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()
Fb = torch.from_numpy(Ftr)
Yb = torch.from_numpy(Ytr)
for ep in range(80):
    opt.zero_grad()
    l = bce(val(Fb), Yb)
    l.backward()
    opt.step()
    if ep % 10 == 0:
        print('ep%d bce=%.4f' % (ep, l.item()), flush=True)
torch.save(val.state_dict(), os.path.join(D, 'value_both27.pt'))

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
