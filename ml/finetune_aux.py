"""Aux-head finetune (Maia-2 style extra supervision).
Warm start: gru_all_v6.pt (engine-order rollout). Shared GRU(14,96) + policy head
+ aux head (96->3): grounded@T+20 (BCE), speed@T+20 (MSE), max-turn@20 (MSE).
Loss = hitbox (friction-first unroll) + 0.2 * masked aux. -> gru_all_v7.pt
Requires: trainA_aux.npz / evalA_aux.npz (build_aux.py) + gru_all_v6.pt.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
LOG = os.path.join(TMP, 'finetune_aux.jsonl')
PROG = os.path.join(TMP, 'finetune_aux_progress.json')
WARM = os.path.join(D, 'gru_all_v6.pt')
MAXV = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HB = 24.0
AUX_W = 0.2


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(14, h, batch_first=True)
        self.head = nn.Linear(h, 2)
        self.aux = nn.Linear(h, 3)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]), self.aux(h[-1])


def maxv_of(xb):
    return (xb[:, -1:, 5:] * torch.from_numpy(MAXV)).sum(-1, keepdim=True)


def apply_friction(v, gr):
    sp = (v ** 2).sum(-1, keepdim=True).sqrt()
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


def unroll(net, xb):
    mv = maxv_of(xb).squeeze(1)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mv
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:]
    hh = h
    traj = []
    for _ in range(66):
        a = net.head(hh[-1]) * mv
        v = apply_friction(v, gr) + a * DT
        p = p + v * DT
        sp = (v ** 2).sum(-1, keepdim=True).sqrt() / mv
        dyn = torch.cat([(v / mv).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static], dim=-1), hh)
        traj.append(p)
    return torch.stack(traj, dim=1), net.head(hh[-1]) * mv


def loss_fn(net, xb, yb, wb, Hb, Ab, Mb):
    mv = maxv_of(xb).squeeze(1)
    traj, ahead = unroll(net, xb)
    l = 0.0
    for hi, h_ in enumerate(HOR):
        tgt = torch.from_numpy(np.array(Hb[:, hi]))
        m = (~torch.isnan(tgt[:, 0:1])).float()
        err = torch.abs(traj[:, h_ - 1] - tgt).amax(dim=1, keepdim=True)
        inside = (err <= HB).float()
        lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
        lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
        l = l + ((lin + lout) * m).sum() / m.sum().clamp_min(1)
    l = l / len(HOR)
    anchor = ((((ahead - yb * mv) / mv) ** 2) * wb).mean() * 0.05
    # aux head on warmup-final hidden (cheap, representation pressure)
    _, h = net.gru(xb)
    au = net.aux(h[-1])
    at = torch.from_numpy(np.array(Ab))
    mk = torch.from_numpy(np.array(Mb))
    laux = (nn.functional.binary_cross_entropy_with_logits(
        au[:, 0:1], at[:, 0:1], reduction='none') * mk[:, 0:1]).mean()
    laux = laux + (((au[:, 1:3] - at[:, 1:3]) ** 2) * mk[:, 1:3]).mean()
    return l + anchor + AUX_W * laux


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


tr = np.load(os.path.join(D, 'trainA.npz'), mmap_mode='r')
ev = np.load(os.path.join(D, 'evalA.npz'), mmap_mode='r')
ta = np.load(os.path.join(D, 'trainA_aux.npz'))
ea = np.load(os.path.join(D, 'evalA_aux.npz'))
Xtr, Ytr, Wtr, Htr = tr['X'], tr['Y'], tr['W'], tr['H']
Xev, Yev, Wev, Hev = ev['X'], ev['Y'], ev['W'], ev['H']
Atr, Mtr = ta['A3'], ta['M']
Aev, Mev = ea['A3'], ea['M']
mtr = (np.abs(Ytr).max(axis=1) <= 40)
mev = (np.abs(Yev).max(axis=1) <= 40)
Xtr, Ytr, Wtr, Htr, Atr, Mtr = Xtr[mtr], Ytr[mtr], Wtr[mtr], Htr[mtr], Atr[mtr], Mtr[mtr]
Xev, Yev, Wev, Hev, Aev, Mev = Xev[mev], Yev[mev], Wev[mev], Hev[mev], Aev[mev], Mev[mev]
print('train', Xtr.shape, 'eval', Xev.shape,
      'auxmask', Mtr.mean(axis=0).round(3), flush=True)

net = Net()
sd = torch.load(WARM, map_location='cpu')
# warm start shared trunk + policy; aux head fresh (not in v6)
net.gru.load_state_dict({k: v for k, v in sd.items() if k.startswith('gru.')})
net.head.load_state_dict({k: v for k, v in sd.items() if k.startswith('head.')})
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
B = 128
N = len(Xtr)
best = 1e18
OUT = os.path.join(D, 'gru_all_v7.pt')
t_start = time.time()
LOG_EVERY = 500
for ep in range(4):
    net.train()
    perm = np.random.permutation(N)
    bl = []
    t0 = time.time()
    for bi, i in enumerate(range(0, N, B)):
        idx = perm[i:i + B]
        xb = torch.from_numpy(np.array(Xtr[idx]))
        yb = torch.from_numpy(np.array(Ytr[idx]))
        wb = torch.from_numpy(np.array(Wtr[idx])).view(-1, 1)
        loss = loss_fn(net, xb, yb, wb, Htr[idx], Atr[idx], Mtr[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        bl.append(loss.item())
        if (bi + 1) % LOG_EVERY == 0:
            ba = np.array(bl)
            log({'type': 'batch', 'ep': ep, 'batch': bi + 1,
                 'train_mean': float(ba.mean()), 'train_std': float(ba.std()),
                 'elapsed_min': round((time.time() - t_start) / 60, 1)})
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        for i in range(0, len(Xev), 512):
            xb = torch.from_numpy(np.array(Xev[i:i + 512]))
            yb = torch.from_numpy(np.array(Yev[i:i + 512]))
            wb = torch.from_numpy(np.array(Wev[i:i + 512])).view(-1, 1)
            el.append(loss_fn(net, xb, yb, wb, Hev[i:i + 512],
                              Aev[i:i + 512], Mev[i:i + 512]).item())
            ew.append(len(xb))
        em = float(np.average(el, weights=ew))
        es = float(np.sqrt(np.average((np.array(el) - em) ** 2, weights=ew)))
    b = np.array(bl)
    rec = {'type': 'epoch', 'ep': ep, 'train_mean': float(b.mean()),
           'train_std': float(b.std()), 'eval_mean': em, 'eval_std': es,
           'ep_sec': round(time.time() - t0, 1),
           'elapsed_min': round((time.time() - t_start) / 60, 1)}
    improved = em < best
    if improved:
        best = em
        torch.save(net.state_dict(), OUT)
    rec['best'] = best
    rec['saved'] = improved
    log(rec)
    with open(PROG, 'w') as f:
        json.dump({'ep': ep, 'of': 4, 'best': best, 'done': False,
                   'elapsed_min': rec['elapsed_min']}, f)
log({'type': 'COMPLETE', 'best': best,
     'elapsed_min': round((time.time() - t_start) / 60, 1)})
with open(PROG, 'w') as f:
    json.dump({'done': True, 'best': best}, f)
