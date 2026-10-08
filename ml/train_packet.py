"""FULL PACKET policy training (generation C, eye frame): head 96->4 = (fm, sm, dyaw, dpitch).

Loss = dense position loss through the yaw-rotating engine rollout
     + input supervision (quadratic on throttle distance, log on direction
       alignment, quadratic on yaw, light quadratic on pitch), all under the
     same difficulty weights w(t).
See inputspace.py (rollout_packet / packet_labels / input_loss) for the spec.

Warm start: accel-space champion trunk (load_packet: input cols zero-init,
head/10, yaw/pitch rows zero). Saves best-eval only. Unique name per line.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inputspace as P

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
SUF = os.environ.get('DS_SUFFIX', 'C')
TAG = 'packet' + os.environ.get('TAG', 'C')
LOG = os.path.join(TMP, TAG + '.log.jsonl')
PROG = os.path.join(TMP, TAG + '_progress.json')
WARM = os.path.join(D, os.environ.get('WARM_CKPT', 'gru_threatsimB.pt'))
OUT = os.path.join(D, 'gru_' + TAG + '.pt')
EPOCHS = int(os.environ.get('EPOCHS', '4'))
LR_TRUNK = float(os.environ.get('LR_TRUNK', '1e-4'))
LR_HEAD = float(os.environ.get('LR_HEAD', '1e-5'))
B = int(os.environ.get('BS', '128'))
MAXSTEPS = int(os.environ.get('MAXSTEPS', '0'))
YAW_SCALE = float(os.environ.get('YAW_SCALE', '16.0'))
LAM_TH = float(os.environ.get('LAM_TH', '0.3'))
LAM_DI = float(os.environ.get('LAM_DI', '0.3'))
LAM_YA = float(os.environ.get('LAM_YA', '0.1'))
PITCH_SCALE = float(os.environ.get('PITCH_SCALE', '8.0'))
LAM_PIT = float(os.environ.get('LAM_PIT', '0.02'))
LAM_VEL = float(os.environ.get('LAM_VEL', '0.1'))  # momentum-training stabilizer
ADDSPEED = os.environ.get('ADDSPEED', '0') == '1'


def load(tag):
    d = np.load(os.path.join(D, tag + SUF + '.npz'), mmap_mode='r')
    k = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    tw = np.load(os.path.join(D, tag + SUF + '_threatwin.npz'))
    dn = np.load(os.path.join(D, tag + SUF + '_dense66.npz'))
    dy = np.load(os.path.join(D, tag + SUF + '_dyaw.npz'))
    Xb = np.array(d['X'])
    TW = np.array(tw['TW'])
    TS = np.array(tw['TS'])
    sol = k[:, 3] == 3
    nsol = int(sol.sum())
    n = min(nsol, TW.shape[0], TS.shape[0], dn['Hd'].shape[0],
            dy['Dy'].shape[0])
    print('%s: soldier keys %d, usable %d' % (tag, nsol, n), flush=True)
    # 22-dim input: 14 base + 6 threat + 2 command slots (zeros in warmup;
    # the rollout fills them with the predicted (fm,sm))
    X = np.concatenate([Xb[sol][:n], TW[:n],
                        np.zeros((n, 66, 2), dtype=np.float32)], axis=2)
    MS = None
    mp = os.path.join(D, tag + SUF + '_mspd.npz')
    if os.path.exists(mp):
        v = np.load(mp)['MS'][:n]
        MS = np.where(np.isfinite(v) & (v >= 50.0), v, 240.0).astype(np.float32)
    Dy = dy['Dy'][:n].astype(np.float32)
    Dp = (dy['Dp'][:n].astype(np.float32) if 'Dp' in dy else
          np.full_like(dy['Dy'][:n], np.nan, dtype=np.float32))
    Fr = dy['Fr'][:n]
    KB = None
    kp = os.path.join(D, tag + SUF + '_knock.npz')
    if os.path.exists(kp):
        KB = np.array(np.load(kp)['KB'][:n, :, 3] > 0.5)
    return (X, TS[:n, -1, :], dn['Hd'][:n], dn['Vd'][:n], MS, Dy, Dp, Fr, KB)


Xtr, Str, Htr, Vtr, Mtr, Dtr, Ptr, Ftr, Ktr = load('train')
Xev, Sev, Hev, Vev, Mev, Dev, Pev, Fev, Kev = load('eval')
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)

net = P.Net(n_in=22, n_out=4)
P.load_packet(net, WARM)
opt = torch.optim.Adam([
    {'params': net.gru.parameters(), 'lr': LR_TRUNK},
    {'params': net.head.parameters(), 'lr': LR_HEAD},
])
KW = dict(yaw_scale_deg=YAW_SCALE, addspeed=ADDSPEED, rule='dot')


def batch_loss(idx, X, S, H, V, M, Dy, Dp, Fr, Kb):
    xb = torch.from_numpy(X[idx])
    sb = torch.from_numpy(S[idx])
    wb = (torch.full((len(idx), 1), 240.0) if M is None
          else torch.from_numpy(M[idx]).reshape(-1, 1))
    mv = (xb[:, -1:, 5:14] * P.MAXV1T).sum(-1, keepdim=True).squeeze(1)
    tr, tau, wh, dyw, dpi = P.rollout_packet(net, xb, sb, wb, **KW)
    # position loss (same hitbox shaping as the champion; tgt nan-safe: NaN*0=NaN)
    tgt = torch.nan_to_num(torch.from_numpy(np.asarray(H[idx])).float(), nan=0.0)
    m = torch.from_numpy(np.asarray(V[idx])).unsqueeze(-1).float()
    err = torch.abs(tr - tgt).amax(dim=-1, keepdim=True)
    inside = (err <= P.HB).float()
    lin = torch.log((err + 1.0) / (P.HB + 1.0)) * inside
    lout = 4.0 * ((err - P.HB) / P.HB).clamp_min(0.0) ** 2 * (1.0 - inside)
    lpos = (((lin + lout) * P.W_T.view(1, -1, 1) * m).sum(dim=1)
            / m.sum(dim=1).clamp_min(1)).mean()
    # input labels from TRUE future positions (inverse dynamics)
    pos = torch.from_numpy(np.asarray(H[idx], dtype=np.float64))
    v1 = torch.cat([pos[:, 0:1] / P.DT,
                    (pos[:, 1:] - pos[:, :-1]) / P.DT], dim=1).float()
    v0 = (xb[:, -1, :2] * mv).unsqueeze(1)
    vel = torch.cat([v0, v1], dim=1).double()
    th, whh, obs = P.packet_labels(vel, xb[:, -1, 3:4].double(), wb.double())
    if Kb is not None:
        # BLAST TICKS: throttle/direction labels are impulse-contaminated, not
        # input - zero their observability (yaw/pitch stay supervised: the view
        # is unaffected by the blast). Recovery ticks keep full supervision.
        obs = obs.float() * (1.0 - torch.from_numpy(
            np.asarray(Kb[idx]).astype(np.float32)).unsqueeze(-1))
    dyh = torch.from_numpy(np.asarray(Dy[idx])).unsqueeze(-1).float()
    dph = torch.from_numpy(np.asarray(Dp[idx])).unsqueeze(-1).float()
    fr = (torch.from_numpy(np.asarray(Fr[idx])).reshape(-1, 1, 1).float()
          * torch.from_numpy(np.asarray(V[idx])).unsqueeze(-1).float())
    linp = P.input_loss(tau, wh, dyw, th, whh, dyh, obs.float(), fr, m,
                        YAW_SCALE, LAM_TH, LAM_DI, LAM_YA,
                        dpi, dph, PITCH_SCALE, LAM_PIT)
    lvel = P.vel_loss(tr, H[idx], V[idx], lam=LAM_VEL)
    return lpos + linp + lvel, float(lpos.detach()), float(linp.detach())


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


N = len(Xtr)
best = 1e18
t_all = time.time()
for ep in range(EPOCHS):
    net.train()
    perm = np.random.permutation(N)
    bl, bp, bi = [], [], []
    t0 = time.time()
    nstep = 0
    for i in range(0, N, B):
        idx = perm[i:i + B]
        opt.zero_grad()
        loss, lp, li = batch_loss(idx, Xtr, Str, Htr, Vtr, Mtr, Dtr, Ptr, Ftr, Ktr)
        if not np.isfinite(float(loss.detach())):
            log({'type': 'SKIP-NONFINITE', 'ep': ep, 'batch': int(i)})
            continue
        loss.backward()
        if not all(torch.isfinite(p.grad).all().item()
                   for p in net.parameters() if p.grad is not None):
            log({'type': 'SKIP-NONFINITE-GRAD', 'ep': ep, 'batch': int(i)})
            continue
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        bl.append(float(loss.detach()))
        bp.append(lp)
        bi.append(li)
        nstep += 1
        if nstep % 100 == 0:
            print('  ep%d step %d/%d loss %.4f (pos %.3f inp %.3f, %.1f min)' % (
                ep, nstep, (N + B - 1) // B, bl[-1], lp, li,
                (time.time() - t_all) / 60), flush=True)
        if MAXSTEPS and nstep >= MAXSTEPS:
            break
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        for i in range(0, len(Xev), 512):
            j = np.arange(i, min(i + 512, len(Xev)))
            l, _, _ = batch_loss(j, Xev, Sev, Hev, Vev, Mev, Dev, Pev, Fev, Kev)
            el.append(float(l))
            ew.append(len(j))
        em = float(np.average(el, weights=ew))
        es = float(np.sqrt(np.average((np.array(el) - em) ** 2, weights=ew)))
    b = np.array(bl)
    rec = {'ep': ep, 'train_mean': float(b.mean()), 'train_std': float(b.std()),
           'pos_mean': float(np.mean(bp)), 'inp_mean': float(np.mean(bi)),
           'eval_mean': em, 'eval_std': es, 'ep_sec': round(time.time() - t0, 1),
           'elapsed_min': round((time.time() - t_all) / 60, 1),
           'cfg': {'yaw': YAW_SCALE, 'lam': [LAM_TH, LAM_DI, LAM_YA],
                   'lr': [LR_TRUNK, LR_HEAD], 'bs': B, 'adds': ADDSPEED}}
    improved = em < best
    if improved:
        best = em
        torch.save(net.state_dict(), OUT)
    rec['best'] = best
    rec['saved'] = improved
    log(rec)
    with open(PROG, 'w') as f:
        json.dump({'ep': ep, 'of': EPOCHS, 'best': best, 'done': False,
                   'elapsed_min': rec['elapsed_min']}, f)
log({'type': 'COMPLETE', 'best': best, 'out': OUT,
     'elapsed_min': round((time.time() - t_all) / 60, 1)})
with open(PROG, 'w') as f:
    json.dump({'done': True, 'best': best, 'out': OUT}, f)
