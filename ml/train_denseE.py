"""Weapon-input dense training (generation E, 21-dim = 14 base + 7 weapon):
identical recipe to train_denseD.py (combat mask, force-mask supervision,
air weight, amax-first NaN guard), PLUS 7 weapon dims: active slot one-hot(3)
+ active class scalar(1) + loadout slot class scalars(3) appended AFTER the
14 base dims (warm-start: trunk 14 cols copied from denseC, 7 new cols
zero-init). OUT: gru_soldier_denseE.pt"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
LOG = os.path.join(TMP, 'trainE.log')
PROG = os.path.join(TMP, 'trainE_progress.json')
SUF = os.environ.get('DS_SUFFIX', 'B')
WARM = os.path.join(D, os.environ.get('WARM_CKPT', 'gru_soldier_denseC.pt'))
AIR_W = float(os.environ.get('AIR_W', '2.0'))
DT = 1 / 66.0
HB = 24.0
W = np.maximum((np.arange(1, 67) / 66.0) ** 1.3718, 0.02).astype(np.float32)
WT = torch.from_numpy(W)
EPOCHS = int(os.environ.get('EPOCHS', '4'))


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(21, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def apply_friction(v, gr):
    # slen-style eps (STATUS doctrine): bare sqrt(0) gives inf gradients
    # (measured torch 2.12.1), which clip_grad_norm turns into NaN weights
    # after step 1 (voided run #1). Forward values change by <1e-6.
    sp = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-12)
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


def unroll(net, xb):
    # 21-dim E layout: [v(2), sp, gr, dy, oh(9), wslot(3), wcls(1), wld(3)].
    # mv/static slice 5:14 ONLY (class one-hot); weapon dims must not leak
    # into the maxspeed norm. Weapon feedback frozen @T (like gr/static):
    # the rollout predicts locomotion, not loadout switches.
    mv = (xb[:, -1:, 5:14] * torch.from_numpy(
        np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32))).sum(-1, keepdim=True)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mv.squeeze(1)
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:14]
    wstat = xb[:, -1:, 14:21]
    hh = h
    traj = []
    for _ in range(66):
        a = net.head(hh[-1]) * mv.squeeze(1)
        v = apply_friction(v, gr) + a * DT
        p = p + v * DT
        sp = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-12) / mv.squeeze(1)
        dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static, wstat], dim=-1), hh)
        traj.append(p)
    return torch.stack(traj, dim=1)


def dense_loss(net, xb, Hb, Vb, w=None, Fm=None):
    traj = unroll(net, xb)  # (B,66,2)
    tgt = torch.from_numpy(np.array(Hb))
    m = torch.from_numpy(np.array(Vb)).unsqueeze(-1).float()
    if Fm is not None:
        # Force supervision mask: horizon ticks corrupted by external impulses
        # (damage knockback transient) carry no reward/penalty. History stays
        # (rows kept, warmup intact); only the loss weight is zeroed.
        m = m * torch.from_numpy(np.array(Fm)).unsqueeze(-1).float()
    # GUARD (2026-10-08, root-caused via anomaly detection): sanitize BEFORE
    # amax. NaN targets on invalid/masked ticks entering amax emit NaN grads
    # (AmaxBackward routes into NaN argmax), clip_grad_norm -> NaN weights.
    diff = (traj - tgt).nan_to_num(0.0)
    err = diff.abs().amax(dim=-1, keepdim=True)  # Chebyshev per tick
    inside = (err <= HB).float()
    lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
    lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
    per = ((lin + lout) * WT.view(1, -1, 1) * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
    if w is not None:
        per = per * torch.from_numpy(np.array(w, dtype=np.float32)).view(-1, 1)
    return per.mean()


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def load(tag):
    # MEMORY-CAPPED: X18 (7GB) and WLD stay on disk as mmap; only per-batch
    # slices enter RAM. Returns memmaps + kept positions, NOT copies.
    d = np.load(os.path.join(D, tag + SUF + '.npz'), mmap_mode='r')
    kk = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    dn = np.load(os.path.join(D, tag + SUF + '_dense66.npz'))
    mk = np.load(os.path.join(D, tag + 'Cmask.npz'))['keep']
    sol = kk[:, 3] == 3
    good = sol & mk
    # NOTE: no row dropping for force ticks (momentum training: history always
    # moves forward per-player, chronological). Force supervision mask applied
    # in loss via Dforce (rows kept, force ticks zero-weighted).
    Xm = d['X']  # mmap (N,66,18): [14 base + wslot(3) + wcls(1)]
    Wm = np.load(os.path.join(D, tag + 'E_wld.npz'), mmap_mode='r')['WLD']  # mmap (N,66,3) uint8
    assert Xm.shape[0] == Wm.shape[0] == kk.shape[0], (Xm.shape, Wm.shape, kk.shape)
    pos = np.where(good)[0]
    sidx = np.where(sol)[0]
    gpos = np.where(good[sidx])[0]
    Hd, Vd = np.array(dn['Hd'][gpos]), np.array(dn['Vd'][gpos])
    # per-row force mask aligned to kept rows (full-key file -> subset)
    Fm = None
    fp = os.path.join(D, tag + 'Dforce.npz')
    if os.path.exists(fp):
        Fall = np.load(fp)['F']
        # Fall is keyed to full Bk; map: good positions in full-key order.
        # Bk keys == SUF keys here (B line); Fall rows align to Bk.
        # Build mask for kept rows: Fall[positions of good in Bk order].
        # good is over SUF keys (=Bk keys), so direct index works.
        Fm = Fall[np.where(good)[0]]
        print(tag, 'force-mask loaded, masked tick frac=%.4f' % float((~Fm).mean() if Fm.size else 0), flush=True)
    return Xm, Wm, pos, Hd, Vd, Fm


def batch_xb(Xm, Wm, pos, idx):
    # assemble 21-dim batch in RAM only: [X18 | WLD/16]
    ii = pos[idx]
    xb18 = torch.from_numpy(np.array(Xm[ii]))
    wld = torch.from_numpy(np.array(Wm[ii])).float() / 16.0
    return torch.cat([xb18, wld], dim=-1)


Xtrm, Wtrm, Ptr, Htr, Vtr, Ftr = load('train')
Xevm, Wevm, Pev, Hev, Vev, Fev = load('eval')
gr_tr = np.array(Xtrm[Ptr, -1, 3])
w_tr = 1.0 + AIR_W * (1.0 - (gr_tr > 0.5).astype(np.float32))
print('train', (len(Ptr), 66, 21), 'eval', (len(Pev), 66, 21), 'AIR_W=%.1f airfrac=%.3f' % (
    AIR_W, (gr_tr < 0.5).mean()), flush=True)
net = Net()
def load_warm_18(net, ckpt):
    """Warm start 21-dim E net from 14-dim denseC: trunk cols 0:14 copied,
    weapon cols 14:21 zero-init (epoch 0 == denseC behavior exactly)."""
    sd = torch.load(ckpt, map_location='cpu')
    out = {}
    for k, v in sd.items():
        if k == 'gru.weight_ih_l0':
            w = torch.zeros_like(net.gru.weight_ih_l0)
            w[:, 0:v.shape[1]] = v
            out[k] = w
        else:
            out[k] = v
    net.load_state_dict(out)
    return net
net = load_warm_18(net, WARM)
opt = torch.optim.Adam(net.parameters(), lr=5e-5)
B = 128
N = len(Ptr)
best = 1e18
OUT = os.path.join(D, 'gru_soldier_denseE.pt')
start_ep = 0
if os.environ.get('RESUME', '0') == '1' and os.path.exists(OUT):
    net.load_state_dict(torch.load(OUT, map_location='cpu'))
    try:
        best = float(json.load(open(PROG))['best'])
    except Exception:
        pass
    start_ep = 1
    print(f'resumed from {OUT} best={best}', flush=True)
t_all = time.time()
for ep in range(start_ep, EPOCHS):
    net.train()
    perm = np.random.permutation(N)
    bl = []
    t0 = time.time()
    for i in range(0, N, B):
        idx = perm[i:i + B]
        xb = batch_xb(Xtrm, Wtrm, Ptr, idx)
        opt.zero_grad()
        loss = dense_loss(net, xb, Htr[idx], Vtr[idx], w_tr[idx],
                          Ftr[idx] if Ftr is not None else None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        bl.append(loss.item())
        if (i // B) % 500 == 0:
            dt = time.time() - t0
            done_frac = (i + B) / N
            print(f'ep{ep} batch {i // B}/{(N + B - 1) // B} loss={bl[-1]:.2f} '
                  f'{dt:.0f}s elapsed, ETA ep ~{dt / max(done_frac, 1e-6):.0f}s', flush=True)
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        for i in range(0, len(Pev), 512):
            xb = batch_xb(Xevm, Wevm, Pev, np.arange(i, min(i + 512, len(Pev))))
            el.append(dense_loss(net, xb, Hev[i:i + 512], Vev[i:i + 512], None,
                                Fev[i:i + 512] if Fev is not None else None).item())
            ew.append(len(xb))
        em = float(np.average(el, weights=ew))
        es = float(np.sqrt(np.average((np.array(el) - em) ** 2, weights=ew)))
    b = np.array(bl)
    rec = {'ep': ep, 'train_mean': float(b.mean()), 'train_std': float(b.std()),
           'eval_mean': em, 'eval_std': es, 'ep_sec': round(time.time() - t0, 1),
           'elapsed_min': round((time.time() - t_all) / 60, 1)}
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
log({'type': 'COMPLETE', 'best': best,
     'elapsed_min': round((time.time() - t_all) / 60, 1)})
with open(PROG, 'w') as f:
    json.dump({'done': True, 'best': best}, f)
