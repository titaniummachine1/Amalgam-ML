"""Combat-masked dense training: trainB arrays filtered by Cmask (damage-anchored
keep: 5s before / 20s after own or any-demo damage, auto-widening via union).
Warm start: best denseB weights. Same dense power-law loss + friction unroll.
SUFFIX env selects line (B). OUT: gru_soldier_denseC.pt"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
LOG = os.path.join(TMP, 'trainC.log')
PROG = os.path.join(TMP, 'trainC_progress.json')
SUF = os.environ.get('DS_SUFFIX', 'B')
WARM = os.path.join(D, os.environ.get('WARM_CKPT', 'gru_soldier_denseB.pt'))
DT = 1 / 66.0
HB = 24.0
W = np.maximum((np.arange(1, 67) / 66.0) ** 1.3718, 0.02).astype(np.float32)
WT = torch.from_numpy(W)
EPOCHS = int(os.environ.get('EPOCHS', '4'))


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(14, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def apply_friction(v, gr):
    sp = (v ** 2).sum(-1, keepdim=True).sqrt()
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


def unroll(net, xb):
    mv = (xb[:, -1:, 5:] * torch.from_numpy(
        np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32))).sum(-1, keepdim=True)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mv.squeeze(1)
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:]
    hh = h
    traj = []
    for _ in range(66):
        a = net.head(hh[-1]) * mv.squeeze(1)
        v = apply_friction(v, gr) + a * DT
        p = p + v * DT
        sp = (v ** 2).sum(-1, keepdim=True).sqrt() / mv.squeeze(1)
        dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static], dim=-1), hh)
        traj.append(p)
    return torch.stack(traj, dim=1)


def dense_loss(net, xb, Hb, Vb):
    traj = unroll(net, xb)
    tgt = torch.from_numpy(np.array(Hb))
    m = torch.from_numpy(np.array(Vb)).unsqueeze(-1).float()
    # GUARD (2026-10-07): see train_denseD.py - NaN targets on invalid ticks
    # poison the masked sum via NaN*0=NaN. Sanitize before masking; valid
    # ticks unaffected.
    err = torch.abs(traj - tgt).amax(dim=-1, keepdim=True).nan_to_num(0.0)
    inside = (err <= HB).float()
    lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
    lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
    per = ((lin + lout) * WT.view(1, -1, 1) * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
    return per.mean()


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def load(tag):
    d = np.load(os.path.join(D, tag + SUF + '.npz'), mmap_mode='r')
    kk = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    dn = np.load(os.path.join(D, tag + SUF + '_dense66.npz'))
    mk = np.load(os.path.join(D, tag + 'Cmask.npz'))['keep']
    sol = kk[:, 3] == 3
    good = sol & mk
    Xs = np.array(d['X'])[good]
    # dense cache rows follow soldier-subset order: map via positions
    sidx = np.where(sol)[0]
    gpos = np.where(good[sidx])[0]
    Hd, Vd = dn['Hd'][gpos], dn['Vd'][gpos]
    return Xs, Hd, Vd


Xtr, Htr, Vtr = load('train')
Xev, Hev, Vev = load('eval')
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)
net = Net()
net.load_state_dict(torch.load(WARM, map_location='cpu'))
opt = torch.optim.Adam(net.parameters(), lr=5e-5)
B = 128
N = len(Xtr)
best = 1e18
OUT = os.path.join(D, 'gru_soldier_denseC.pt')
t_all = time.time()
for ep in range(EPOCHS):
    net.train()
    perm = np.random.permutation(N)
    bl = []
    t0 = time.time()
    for i in range(0, N, B):
        idx = perm[i:i + B]
        xb = torch.from_numpy(np.array(Xtr[idx]))
        opt.zero_grad()
        loss = dense_loss(net, xb, Htr[idx], Vtr[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        bl.append(loss.item())
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        for i in range(0, len(Xev), 512):
            xb = torch.from_numpy(np.array(Xev[i:i + 512]))
            el.append(dense_loss(net, xb, Hev[i:i + 512], Vev[i:i + 512]).item())
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
