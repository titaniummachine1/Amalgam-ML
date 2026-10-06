"""Known-forces accel policy training: GRU(23,96) + head(96,2), warm start.

Physics (2-pass, input last): friction -> known impulse -> integrate
(+ collide hook, identity until the wall grid lands) -> policy accel.
Loss = dense hitbox loss x w2 trainability weights. Knockback WITH known
force trains; trash (unknown-force cap violations, NaNs) is zeroed.
Saves best-eval only. Unique name per line.
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
TAG = 'forces' + os.environ.get('TAG', 'A')
LOG = os.path.join(TMP, TAG + '.log.jsonl')
PROG = os.path.join(TMP, TAG + '_progress.json')
WARM = os.path.join(D, os.environ.get('WARM_CKPT', 'gru_threatsimB.pt'))
OUT = os.path.join(D, 'gru_' + TAG + '.pt')
EPOCHS = int(os.environ.get('EPOCHS', '4'))
LR = float(os.environ.get('LR', '1e-4'))
B = int(os.environ.get('BS', '128'))
MAXSTEPS = int(os.environ.get('MAXSTEPS', '0'))
K_IMP = float(os.environ.get('K_IMP', '0.0'))  # 0 = feature-only until K_fit measured
USE_W2 = os.environ.get('USE_W2', '1') == '1'
LAM_VEL = float(os.environ.get('LAM_VEL', '0.1'))  # velocity stabilizer


def load(tag):
    d = np.load(os.path.join(D, tag + SUF + '.npz'), mmap_mode='r')
    k = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    tw = np.load(os.path.join(D, tag + SUF + '_threatwin.npz'))
    dn = np.load(os.path.join(D, tag + SUF + '_dense66.npz'))
    kn = np.load(os.path.join(D, tag + SUF + '_knock.npz'))
    Xb = np.array(d['X'])
    TW = np.array(tw['TW'])
    TS = np.array(tw['TS'])
    sol = k[:, 3] == 3
    nsol = int(sol.sum())
    ts = np.load(os.path.join(D, tag + SUF + '_tsf.npz'))
    n = min(nsol, TW.shape[0], TS.shape[0], dn['Hd'].shape[0], kn['KB'].shape[0],
            ts['TSF'].shape[0])
    print('%s: soldier keys %d, usable %d' % (tag, nsol, n), flush=True)
    X = np.concatenate([Xb[sol][:n], TW[:n], np.array(kn['KBW'][:n]),
                        np.array(ts['TSF'][:n])], axis=2)
    assert X.shape[2] == 20 + P.NEX, X.shape
    W2 = None
    wp = os.path.join(D, tag + SUF + '_w2.npz')
    if USE_W2 and os.path.exists(wp):
        W2 = np.array(np.load(wp)['W2'][:n]).astype(np.float32)
        print('%s w2: mean %.3f frac0 %.4f' % (tag, W2.mean(), (W2 == 0).mean()),
              flush=True)
    return (X, TS[:n, -1, :], dn['Hd'][:n], dn['Vd'][:n],
            np.array(kn['KB'][:n]), W2)


Xtr, Str, Htr, Vtr, Ktr, Wtr = load('train')
Xev, Sev, Hev, Vev, Kev, Wev = load('eval')
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)

net = P.Net(n_in=20 + P.NEX, n_out=2)
P.load_forces(net, WARM)
opt = torch.optim.Adam(net.parameters(), lr=LR)


def batch_loss(idx, X, S, H, V, K, W2):
    return P.forces_loss(
        net, torch.from_numpy(X[idx]), torch.from_numpy(S[idx]),
        torch.from_numpy(K[idx]), H[idx], V[idx],
        None if W2 is None else torch.from_numpy(W2[idx]).unsqueeze(-1),
        k_imp=K_IMP, lam_vel=LAM_VEL)


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
        loss = batch_loss(idx, Xtr, Str, Htr, Vtr, Ktr, Wtr)
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
        nstep += 1
        if nstep % 100 == 0:
            print('  ep%d step %d/%d loss %.4f (%.1f min)' % (
                ep, nstep, (N + B - 1) // B, bl[-1], (time.time() - t_all) / 60),
                flush=True)
        if MAXSTEPS and nstep >= MAXSTEPS:
            break
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        for i in range(0, len(Xev), 512):
            j = np.arange(i, min(i + 512, len(Xev)))
            el.append(float(batch_loss(j, Xev, Sev, Hev, Vev, Kev, Wev)))
            ew.append(len(j))
        em = float(np.average(el, weights=ew))
        es = float(np.sqrt(np.average((np.array(el) - em) ** 2, weights=ew)))
    b = np.array(bl)
    rec = {'ep': ep, 'train_mean': float(b.mean()), 'train_std': float(b.std()),
           'eval_mean': em, 'eval_std': es, 'ep_sec': round(time.time() - t0, 1),
           'elapsed_min': round((time.time() - t_all) / 60, 1),
           'cfg': {'k_imp': K_IMP, 'w2': USE_W2, 'lr': LR, 'bs': B}}
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
