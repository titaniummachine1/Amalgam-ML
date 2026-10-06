"""Verdict for generation C (eye frame). Protocol mirrors hit_eval_final.py but on
C data (evalC soldier, |Y|<=40, RandomState(1) 2500 rows): coast, accel champion
(gru_threatsimC.pt), packet model (gru_packetC.pt), plus:

  sign  teacher-forced yaw test: drive rollout_packet's psi with the TRUE
        accumulated eye deltas (+psi) vs negated (-psi) vs frozen (0). The
        correct rotation convention must win clearly; if -psi wins, the
        forward/right mapping is mirrored and rollout_packet must be fixed.
        Uses the champion's (fm,sm) direction with full throttle so only the
        rotation is under test.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inputspace as P

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
SUF = 'C'
HOR = [7, 13, 20, 33, 66]
HITR = 24.0
MAXV = 240.0
CK = os.environ.get('CKPT', 'gru_packetC.pt')
CHAMP = os.environ.get('CHAMP', 'gru_threatsimC.pt')
YAW_SCALE = float(os.environ.get('YAW_SCALE', '16.0'))

d = np.load(os.path.join(D, 'eval' + SUF + '.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'eval' + SUF + 'k.npz'))['K']
tw = np.load(os.path.join(D, 'eval' + SUF + '_threatwin.npz'))
dn = np.load(os.path.join(D, 'eval' + SUF + '_dense66.npz'))
dy = np.load(os.path.join(D, 'eval' + SUF + '_dyaw.npz'))
Xb, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
TW, TS = np.array(tw['TW']), np.array(tw['TS'])
sol = k[:, 3] == 3
Xb, TW, TS = Xb[sol], TW, TS
X20 = np.concatenate([Xb, TW], axis=2)
S0 = TS[:, -1, :]
m = (np.abs(Y[sol]).max(axis=1) <= 40)
Xb, X20, S0 = Xb[m], X20[m], S0[m]
H = H[sol][m]
Hd = dn['Hd'][m]
Dy = dy['Dy'][m]
active = (TW[m][:, -1, 3] > 0) | (TW[m][:, -1, 4] > 0)
_z = np.load(os.path.join(D, 'eval' + SUF + '_mspd.npz'))
MS = np.where(np.isfinite(_z['MS'][m]), _z['MS'][m], 240.0)
MS = np.where(MS >= 50.0, MS, 240.0).astype(np.float32)

idx = np.random.RandomState(1).choice(len(Xb), min(2500, len(Xb)), replace=False)
Xb, X20, S0, H, Hd, Dy, active, MS = (a[idx] for a in (Xb, X20, S0, H, Hd, Dy, active, MS))
Y = Y[sol][m][idx]
v0 = Xb[:, -1, :2] * MAXV
print('n =', len(Xb), 'threat-active frac = %.3f' % active.mean(), flush=True)


def accel_roll(net, X, S):
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


def packet_roll(net, X, S, W, **kw):
    X22 = np.concatenate([X, np.zeros((len(X), 66, 2), dtype=np.float32)], axis=2)
    outs = []
    with torch.no_grad():
        for b in range(0, len(X22), 128):
            tr, _, _, _, _ = P.rollout_packet(
                net, torch.from_numpy(X22[b:b + 128]), torch.from_numpy(S[b:b + 128]),
                torch.from_numpy(W[b:b + 128]).reshape(-1, 1), **kw)
            outs.append(tr.numpy())
    return np.concatenate(outs, 0)


def forced_roll(net, X, S, W, psi_deg, **kw):
    """Teacher-forced rotation: the PACKET model's own (fm,sm) with PRESCRIBED
    psi. psi_deg (B,66) = accumulated yaw through T+t-1 per step t (cumsum of
    per-tick deltas, shifted right, 0 prepended). Pass +true for the upper
    bound on yaw modeling, -true to check the sign convention (must be clearly
    worse), zeros for no-rotation. Replaces the old mixed-frame test (champion
    direction in velocity frame rotated by eye yaw), which was broken."""
    X22 = np.concatenate([X, np.zeros((len(X), 66, 2), dtype=np.float32)], axis=2)
    outs = []
    with torch.no_grad():
        for b in range(0, len(X22), 128):
            tr, _, _, _, _ = P.rollout_packet(
                net, torch.from_numpy(X22[b:b + 128]), torch.from_numpy(S[b:b + 128]),
                torch.from_numpy(W[b:b + 128]).reshape(-1, 1),
                psi_force=torch.from_numpy(psi_deg[b:b + 128]).float(), **kw)
            outs.append(tr.numpy())
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
        print('  %-10s %s' % (name, ' '.join(line)), flush=True)


rows = [('coast', None)]
champ = P.Net()
champ.load_state_dict(torch.load(os.path.join(D, CHAMP), map_location='cpu'))
champ.eval()
print('rolling C accel champion...', flush=True)
rows.append(('C-champ', accel_roll(champ, X20, S0)))

if os.path.exists(os.path.join(D, CK)):
    net = P.Net(n_in=22, n_out=4)
    net.load_state_dict(torch.load(os.path.join(D, CK), map_location='cpu'))
    net.eval()
    print('rolling %s...' % CK, flush=True)
    rows.append(('packet', packet_roll(net, X20, S0, MS, yaw_scale_deg=YAW_SCALE)))
    # sign test on the PACKET model (same frame throughout): teacher-forced true
    # psi must beat predicted psi (else yaw modeling is the bottleneck) and must
    # beat -psi clearly (else the rotation convention is mirrored).
    dy0 = np.concatenate([np.zeros_like(Dy[:, 0:1]), np.nan_to_num(Dy, nan=0.0)[:, :-1]], axis=1)
    psi_step = np.cumsum(dy0, axis=1)
    for tag, ps in [('+psi', psi_step), ('-psi', -psi_step),
                    ('psi0', np.zeros_like(psi_step))]:
        print('rolling forced %s...' % tag, flush=True)
        rows.append(('force' + tag, forced_roll(
            net, X20, S0, MS, ps, yaw_scale_deg=YAW_SCALE)))
else:
    print('(no %s yet)' % CK, flush=True)

for mask, lab in [(np.ones(len(Xb), dtype=bool), 'ALL'), (active, 'THREAT-ACTIVE')]:
    report(mask, lab, rows)
