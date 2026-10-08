"""INPUT-SPACE policy training: the head outputs the usercmd (forwardmove, sidemove)
and the rollout turns that into acceleration with the engine's Accelerate math.

  head 96->2, tanh-free RADIAL bound (inputspace.policy_out): the direction is kept
  exactly and only the magnitude is capped, so a head warm-started from the
  accel-space champion (head/10) starts at the champion's own rollout. tanh was
  measured and rejected: it compresses the working range (2% low at the median
  accel, 19% at p90).

  rollout per step:  wishdir = (fm,sm)/|(fm,sm)|
                     wishspeed = MaxSpeed * min(|(fm,sm)|,1)
                     v = sv_friction(v)  then  v += wishdir * sv_accelerate*dt*wishspeed
                     p += v*dt ;  previous throttle fed back into the dyn slot
  (the add-speed early-out `wishspeed - |v|` is disabled: it is measurably
  incompatible with this dataset, see inputspace.accelerate)

  loss: the champion's dense hitbox loss on the SAME targets. No single-step anchor
  on (fm,sm) - inputs are unobservable at steady state, so any such target is
  wrongly supervised. There is also no dyaw aux: the demos carry no view angle
  (m_angEyeAngles is local-exclusive and never arrives from SourceTV), so yaw has
  no ground truth at all.

Warm start: gru_threatsimB.pt (champion). Saves best-eval only, unique name.
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
SUF = os.environ.get('DS_SUFFIX', 'B')
TAG = 'inputspace' + os.environ.get('TAG', 'A')
LOG = os.path.join(TMP, TAG + '.log.jsonl')
PROG = os.path.join(TMP, TAG + '_progress.json')
WARM = os.path.join(D, os.environ.get('WARM_CKPT', 'gru_threatsimB.pt'))
OUT = os.path.join(D, 'gru_' + TAG + '.pt')
EPOCHS = int(os.environ.get('EPOCHS', '4'))
LR_TRUNK = float(os.environ.get('LR_TRUNK', '1e-4'))
LR_HEAD = float(os.environ.get('LR_HEAD', '1e-5'))   # head is scaled 1/10, so its
                                                     # output-space step is 1e-4
B = int(os.environ.get('BS', '128'))
ADDSPEED = os.environ.get('ADDSPEED', '0') == '1'
FEEDBACK = os.environ.get('FEEDBACK', '1') == '1'
USE_MSPD = os.environ.get('USE_MSPD', '1') == '1'
MAXSTEPS = int(os.environ.get('MAXSTEPS', '0'))   # smoke-test escape hatch
TRUNC = int(os.environ.get('TRUNC', '0'))        # 0 = use every usable row
KW = dict(addspeed=ADDSPEED, feedback=FEEDBACK, mode='radial')


def load(tag):
    d = np.load(os.path.join(D, tag + SUF + '.npz'), mmap_mode='r')
    k = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    tw = np.load(os.path.join(D, tag + SUF + '_threatwin.npz'))
    Xb = np.array(d['X'])
    TW = np.array(tw['TW'])
    TS = np.array(tw['TS'])
    sol = k[:, 3] == 3
    nsol = int(sol.sum())
    # dense targets: prefer the soldier-aligned cache; fall back to the full-key one
    dp = os.path.join(D, tag + SUF + '_dense66.npz')
    dn = np.load(dp)
    if dn['Hd'].shape[0] != nsol:
        alt = os.path.join(D, tag + SUF + '_dense66all.npz')
        if os.path.exists(alt):
            z = np.load(alt)
            if z['Hd'].shape[0] == len(sol):
                dn = {'Hd': z['Hd'][sol], 'Vd': z['Vd'][sol]}
    MS = None
    mp = os.path.join(D, tag + SUF + '_mspd.npz')
    if USE_MSPD and os.path.exists(mp):
        v = np.load(mp)['MS']
        # mspd=1 is what dead / not-yet-spawned rows report: not a real cap
        MS = np.where(np.isfinite(v) & (v >= 50.0), v, 240.0).astype(np.float32)
    # Caches can be SHORTER than the key file (the 2026-09-30 data expansion grew
    # the keys after the train threat/dense caches were built). The old caches are
    # a bit-identical PREFIX of the new ones, so the common prefix is a valid,
    # data-matched training set - and exactly the champion's, at 189565 rows.
    n = min(nsol, TW.shape[0], TS.shape[0], dn['Hd'].shape[0], dn['Vd'].shape[0],
            0 if MS is None else len(MS))
    if TRUNC:
        n = min(n, TRUNC)
    stale = ' STALE-CACHE' if n < nsol else ''
    print('%s: soldier keys %d, usable %d%s' % (tag, nsol, n, stale), flush=True)
    X = np.concatenate([Xb[sol][:n], TW[:n]], axis=2)
    return (X, TS[:n, -1, :], dn['Hd'][:n], dn['Vd'][:n],
            None if MS is None else MS[:n])


Xtr, Str, Htr, Vtr, Mtr = load('train')
Xev, Sev, Hev, Vev, Mev = load('eval')
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)
if Mtr is not None:
    print('train wcap!=240 frac %.4f' % (Mtr != 240.0).mean(), flush=True)
assert len(Xtr) == len(Htr) and len(Xev) == len(Hev)

net = P.Net()
net.load_reparam(WARM)
opt = torch.optim.Adam([
    {'params': net.gru.parameters(), 'lr': LR_TRUNK},
    {'params': net.head.parameters(), 'lr': LR_HEAD},
])


def batch_loss(idx, X, S, H, V, M):
    xb = torch.from_numpy(X[idx])
    sb = torch.from_numpy(S[idx])
    wb = None if M is None else torch.from_numpy(M[idx])
    return P.dense_loss(net, xb, sb, H[idx], V[idx], wcap=wb, **KW)


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
    bl = []
    t0 = time.time()
    nstep = 0
    for i in range(0, N, B):
        idx = perm[i:i + B]
        opt.zero_grad()
        loss = batch_loss(idx, Xtr, Str, Htr, Vtr, Mtr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        bl.append(loss.item())
        nstep += 1
        if nstep % 100 == 0:
            print('  ep%d step %d/%d loss %.4f (%.1f min)' % (
                ep, nstep, (N + B - 1) // B, loss.item(), (time.time() - t_all) / 60),
                flush=True)
        if MAXSTEPS and nstep >= MAXSTEPS:
            break
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        for i in range(0, len(Xev), 512):
            el.append(batch_loss(np.arange(i, min(i + 512, len(Xev))), Xev, Sev,
                                 Hev, Vev, Mev).item())
            ew.append(min(512, len(Xev) - i))
        em = float(np.average(el, weights=ew))
        es = float(np.sqrt(np.average((np.array(el) - em) ** 2, weights=ew)))
    b = np.array(bl)
    rec = {'ep': ep, 'train_mean': float(b.mean()), 'train_std': float(b.std()),
           'eval_mean': em, 'eval_std': es, 'ep_sec': round(time.time() - t0, 1),
           'elapsed_min': round((time.time() - t_all) / 60, 1),
           'cfg': {'addspeed': ADDSPEED, 'feedback': FEEDBACK, 'mspd': USE_MSPD,
                   'lr': [LR_TRUNK, LR_HEAD], 'bs': B}}
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
