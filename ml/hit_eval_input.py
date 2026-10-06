"""Verdict for the INPUT-SPACE (fm,sm) policy. Exact protocol of hit_eval_final.py
(evalB soldier, |Y|<=40 mask, RandomState(1) 2500 rows, threat-active subset) so
the numbers are directly comparable to the ledger:

  coast 92.5/73.5/51.7/30.0/12.4   sim(champion) 97.8/87.4/60.0/27.4/12.5

Also runs three diagnostics:
  reparam  the champion head/10 driven through the engine Accelerate rollout. It
           MUST land on the champion's numbers - that is the proof the physics
           module is a faithful re-encoding and not a different dynamics.
  cos      cosine between the decoded wishdir and the OBSERVED pure-input accel
           direction (the label). The packet policy is only "real inputs" if this
           is high; a low value means it found its own parameterisation.
  cap      how often the engine clamps (wishspeed at the ceiling / no headroom),
           i.e. how much of the rollout is physics-bound rather than free.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inputspace as P

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
HITR = 24.0
MAXV = 240.0
CK = os.environ.get('CKPT', 'gru_inputspaceA.pt')
WARM = os.environ.get('WARM', 'gru_threatsimB.pt')

d = np.load(os.path.join(D, 'evalB.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalBk.npz'))['K']
tw = np.load(os.path.join(D, 'evalB_threatwin.npz'))
Xb, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
TW, TS = np.array(tw['TW']), np.array(tw['TS'])
sol = k[:, 3] == 3
Xb, TW, TS = Xb[sol], TW, TS
X20 = np.concatenate([Xb, TW], axis=2)
S0 = TS[:, -1, :]
m = (np.abs(Y[sol]).max(axis=1) <= 40)
Xb, X20, S0 = Xb[m], X20[m], S0[m]
H = H[sol][m]
active = (TW[m][:, -1, 3] > 0) | (TW[m][:, -1, 4] > 0)

idx = np.random.RandomState(1).choice(len(Xb), min(2500, len(Xb)), replace=False)
Xb, X20, S0, H, active = Xb[idx], X20[idx], S0[idx], H[idx], active[idx]
Y = Y[sol][m][idx]
MSp = None
_ms = os.path.join(D, 'evalB_mspd.npz')
if os.path.exists(_ms):
    z = np.load(_ms)
    MS = z['MS'][m][idx]          # already soldier-only, same order as Xb
    # mspd=1 appears on dead/not-yet-spawned rows (0.1%): not a real cap
    MSp = np.where(np.isfinite(MS) & (MS >= 50.0), MS, 240.0).astype(np.float32)
    print('wcap from m_flMaxspeed: frac != 240 = %.4f' % (MSp != 240.0).mean(), flush=True)
v0 = Xb[:, -1, :2] * MAXV
print('n =', len(Xb), 'threat-active frac = %.3f' % active.mean(), flush=True)


def roll20(net, X, S, W=None, **kw):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            outs.append(P.rollout(net, torch.from_numpy(X[b:b + 128]),
                                  torch.from_numpy(S[b:b + 128]), wcap=None if W is None
                                  else torch.from_numpy(W[b:b + 128]), **kw).numpy())
    return np.concatenate(outs, axis=0)

def accel_rollout(net, X, S):
    """Champion dynamics (free 2D accel) for the reference row."""
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = torch.from_numpy(X[b:b + 128])
            sb = torch.from_numpy(S[b:b + 128])
            _, h = net.gru(xb)
            mv = (xb[:, -1:, 5:14] * P.MAXV1T).sum(-1, keepdim=True)
            v = xb[:, -1, :2] * mv.squeeze(1)
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            static = xb[:, -1:, 5:14]
            q = sb[:, 0:2].clone()
            vth = sb[:, 2:4].clone()
            present = sb[:, 4:5]
            nspd = torch.clamp((vth ** 2).sum(-1, keepdim=True).sqrt() / 3000.0, 0.0, 1.0)
            tf = xb[:, -1:, 14:20]
            cnt, mem = tf[:, 0, 3:4], tf[:, 0, 4:5]
            hh, hs = h, []
            for _ in range(66):
                a = net.head(hh[-1]) * mv.squeeze(1)
                v = P.friction(v, gr) + a * P.DT
                p = p + v * P.DT
                q = q + vth * P.DT
                rel = q - p
                dist = rel.norm(dim=1, keepdim=True).clamp_min(1e-6)
                ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
                ang = torch.atan2(rel[:, 1:2], rel[:, 0:1])
                infov = (ang.abs() < (np.pi / 4)).float()
                appr = -((vth - v) * rel).sum(-1, keepdim=True) / dist
                tti = torch.where(appr > 50.0, torch.clamp(dist / appr / 5.0, 0.0, 1.0),
                                  torch.ones_like(dist))
                live = torch.cat([ndist, nspd, infov, cnt, mem, tti], -1).unsqueeze(1)
                thr = present.unsqueeze(1) * live + (1 - present.unsqueeze(1)) * tf
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / mv.squeeze(1)
                dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                                 gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1), thr], -1)
                _, hh = net.gru(torch.cat([dyn, static], -1), hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, 1).numpy())
    return np.concatenate(outs, 0)


def report(mask, label, rows):
    if mask.sum() < 100:
        return
    print('--- %s (n=%d) ---' % (label, mask.sum()), flush=True)
    for name, arr in rows:
        line = []
        for hi, h_ in enumerate(HOR):
            tgt = H[mask][:, hi]
            mm = ~np.isnan(tgt[:, 0])
            err = (np.abs((v0[mask] * (h_ * P.DT))[mm] - tgt[mm]).max(axis=1)
                   if arr is None else
                   np.abs(arr[mask][mm][:, h_ - 1] - tgt[mm]).max(axis=1))
            line.append('%d:%.1f%%' % (h_, (err < HITR).mean() * 100))
        print('  %-9s %s' % (name, ' '.join(line)), flush=True)


rows = [('coast', None)]
raw = P.Net()
raw.load_state_dict(torch.load(os.path.join(D, WARM), map_location='cpu'))
raw.eval()
print('rolling accel-space champion (reference)...', flush=True)
rows.append(('champ', accel_rollout(raw, X20, S0)))
rp = P.Net()
rp.load_reparam(os.path.join(D, WARM))
rp.eval()
rp2 = P.Net()
rp2.load_reparam(os.path.join(D, WARM), reparam=2.0)
rp2.eval()
CFG = [('240cap+as-mag', None, dict(addspeed=True, rule='mag')),
       ('240cap+as-dot', None, dict(addspeed=True, rule='dot')),
       ('MS +as-dot', MSp, dict(addspeed=True, rule='dot')),
       ('MS -as', MSp, dict(addspeed=False))]
for tag, W, kw in CFG:
    print('rolling reparam %s...' % tag, flush=True)
    rows.append(('rp' + tag, roll20(rp, X20, S0, W=W, mode='radial', feedback=True, **kw)))
print('rolling reparam/2 full-throttle dot...', flush=True)
rows.append(('rp/2-dot', roll20(rp2, X20, S0, W=MSp, mode='radial',
                               feedback=True, addspeed=True, rule='dot')))
print('rolling accel harness (raw champion through P.rollout)...', flush=True)
rows.append(('harness', roll20(raw, X20, S0, mode='accel')))
if os.path.exists(os.path.join(D, CK)):
    net = P.Net()
    net.load_state_dict(torch.load(os.path.join(D, CK), map_location='cpu'))
    net.eval()
    print('rolling %s...' % CK, flush=True)
    rows.append(('input', roll20(net, X20, S0)))
else:
    net = None
    print('(no %s yet)' % CK, flush=True)

for mask, lab in [(np.ones(len(Xb), dtype=bool), 'ALL'), (active, 'THREAT-ACTIVE')]:
    report(mask, lab, rows)

# ---- policy diagnostics -----------------------------------------------------
if net is not None:
    net.eval()
    C, THR, NOACC, SAT = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(X20), 128):
            xb = torch.from_numpy(X20[b:b + 128])
            sb = torch.from_numpy(S0[b:b + 128])
            _, h = net.gru(xb)
            mv = (xb[:, -1:, 5:14] * P.MAXV1T).sum(-1, keepdim=True)
            v = xb[:, -1, :2] * mv.squeeze(1)
            gr = xb[:, -1, 3:4]
            hh = h
            for _ in range(66):
                o = torch.tanh(net.head(hh[-1]))
                mag = o.norm(dim=1, keepdim=True)
                wh = o / mag.clamp_min(1e-6)
                w = mv * torch.clamp(mag, max=1.0)
                vf = P.friction(v, gr)
                sp = (vf ** 2).sum(-1, keepdim=True).sqrt()
                add = torch.clamp(w - sp, min=0.0)
                dv = torch.minimum(add, P.ACCEL * P.DT * w)
                C.append(wh)
                THR.append(torch.clamp(mag, max=1.0).squeeze(1))
                SAT.append((mag >= 1.0).squeeze(1).float())
                NOACC.append((add <= 0).squeeze(1).float())
                v = vf + wh * dv
                p_ = torch.zeros_like(v)  # position unused here
                spn = (v ** 2).sum(-1, keepdim=True).sqrt() / mv.squeeze(1)
                fb = torch.clamp(mag, max=1.0)
                dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), spn.unsqueeze(1),
                                 gr.unsqueeze(1), fb,
                                 xb[:, -1:, 14:20]], dim=-1)
                _, hh = net.gru(torch.cat([dyn, xb[:, -1:, 5:14]], -1), hh)
    WH = torch.cat(C).numpy()
    ay, ax = Y[:, 1], Y[:, 0]
    n = np.linalg.norm(np.stack([ax, ay]), axis=0)
    keep = n > 0.5
    cos = (WH[0][keep, 0] * ax[keep] + WH[0][keep, 1] * ay[keep]) / n[keep]
    print('\npolicy diagnostics (step 1):')
    print('  cos(decoded wishdir, observed accel dir) mean %.3f  |accel|>1: %.3f' % (
        cos.mean(), keep.mean()))
    print('  throttle |(fm,sm)| mean %.3f   at ceiling(>=1) %.3f' % (
        torch.cat(THR).mean().item(), torch.cat(SAT).mean().item()))
    print('  engine gives ZERO accel (speed>=wishspeed) on %.3f of steps' % torch.cat(NOACC).mean().item())
