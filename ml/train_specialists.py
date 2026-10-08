"""Nine class specialists, warm-started from the all-class champion (gru_all_v5.pt).
Fixes two things at once:
  1. horizon weighting: hard horizons (500/1000ms) get more gradient than saturated
     ones (100ms at 96.5% contributes ~nothing) -> fixes the 500ms deficit vs coast.
  2. thin classes (medic 18k, engy 11k, pyro 27k) transfer from the all-class net
     instead of training from scratch on nothing.
Loss: hitbox (log inside 24hu / 4x quad outside) per horizon, weighted. No single-step
anchor (user: single-tick objective must not pull focus).
Loops all 9 classes, saves gru_spec_<class>.pt, logs JSONL + progress file.
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
LOG = os.path.join(TMP, 'specialists.jsonl')
PROG = os.path.join(TMP, 'specialists_progress.json')
CHAMP = os.path.join(D, 'gru_all_v5.pt')
MAXV = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)
CLSN = {1: 'scout', 2: 'sniper', 3: 'soldier', 4: 'demo', 5: 'medic',
        6: 'heavy', 7: 'pyro', 8: 'spy', 9: 'engy'}
HOR = [7, 13, 20, 33, 66]
HW = np.array([1.0, 1.0, 1.5, 3.0, 3.0], dtype=np.float32)  # hard-horizon emphasis
HW = HW / HW.sum()
DT = 1 / 66.0
HB = 24.0
EPOCHS = 3


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(14, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


tr = np.load(os.path.join(D, 'trainA.npz'), mmap_mode='r')
ev = np.load(os.path.join(D, 'evalA.npz'), mmap_mode='r')
ktr = np.load(os.path.join(D, 'trainAk.npz'))['K']
kev = np.load(os.path.join(D, 'evalAk.npz'))['K']
Xtr_all, Ytr_all, Wtr_all, Htr_all = tr['X'], tr['Y'], tr['W'], tr['H']
Xev_all, Yev_all, Wev_all, Hev_all = ev['X'], ev['Y'], ev['W'], ev['H']
mtr = (np.abs(Ytr_all).max(axis=1) <= 40)
mev = (np.abs(Yev_all).max(axis=1) <= 40)
ctr = ktr[:, 3]   # full-length class arrays; mask applied per class below
cev = kev[:, 3]


def maxv_of(xb):
    return (xb[:, -1:, 5:] * torch.from_numpy(MAXV)).sum(-1, keepdim=True)


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
        v = v + a * DT
        p = p + v * DT
        sp = (v ** 2).sum(-1, keepdim=True).sqrt() / mv
        dyn = torch.cat([(v / mv).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static], dim=-1), hh)
        traj.append(p)
    return torch.stack(traj, dim=1)


def hitbox_loss(net, xb, Hb):
    traj = unroll(net, xb)
    l = 0.0
    for hi, h_ in enumerate(HOR):
        tgt = torch.from_numpy(np.array(Hb[:, hi]))
        m = (~torch.isnan(tgt[:, 0:1])).float()
        err = torch.abs(traj[:, h_ - 1] - tgt).amax(dim=1, keepdim=True)
        inside = (err <= HB).float()
        lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
        lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
        l = l + HW[hi] * ((lin + lout) * m).sum() / m.sum().clamp_min(1)
    return l


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


t_all = time.time()
for cls in range(1, 10):
    name = CLSN[cls]
    mtr_c = mtr & (ctr == cls)
    mev_c = mev & (cev == cls)
    n_tr, n_ev = int(mtr_c.sum()), int(mev_c.sum())
    log({'type': 'start', 'cls': cls, 'name': name, 'n_train': n_tr, 'n_eval': n_ev})
    if n_tr < 500 or n_ev < 100:
        log({'type': 'skip', 'cls': cls, 'name': name, 'why': 'too few windows'})
        continue
    Xtr, Ytr, Wtr, Htr = Xtr_all[mtr_c], Ytr_all[mtr_c], Wtr_all[mtr_c], Htr_all[mtr_c]
    Xev, Yev, Wev, Hev = Xev_all[mev_c], Yev_all[mev_c], Wev_all[mev_c], Hev_all[mev_c]
    net = Net()
    net.load_state_dict(torch.load(CHAMP, map_location='cpu'))  # warm start
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    B = 128
    N = len(Xtr)
    best = 1e18
    out = os.path.join(D, f'gru_spec_{name}.pt')
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(N)
        bl = []
        t0 = time.time()
        for i in range(0, N, B):
            idx = perm[i:i + B]
            xb = torch.from_numpy(np.array(Xtr[idx]))
            opt.zero_grad()
            loss = hitbox_loss(net, xb, Htr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            bl.append(loss.item())
        net.eval()
        with torch.no_grad():
            el, ew = [], []
            for i in range(0, len(Xev), 512):
                xb = torch.from_numpy(np.array(Xev[i:i + 512]))
                el.append(hitbox_loss(net, xb, Hev[i:i + 512]).item())
                ew.append(len(xb))
            em = float(np.average(el, weights=ew))
            es = float(np.sqrt(np.average((np.array(el) - em) ** 2, weights=ew)))
        b = np.array(bl)
        rec = {'type': 'epoch', 'cls': cls, 'name': name, 'ep': ep,
               'train_mean': float(b.mean()), 'train_std': float(b.std()),
               'eval_mean': em, 'eval_std': es, 'ep_sec': round(time.time() - t0, 1),
               'elapsed_min': round((time.time() - t_all) / 60, 1)}
        improved = em < best
        if improved:
            best = em
            torch.save(net.state_dict(), out)
        rec['best'] = best
        rec['saved'] = improved
        log(rec)
        with open(PROG, 'w') as f:
            json.dump({'cls': cls, 'name': name, 'ep': ep, 'of': EPOCHS,
                       'best': best, 'n_train': n_tr, 'done': False,
                       'elapsed_min': rec['elapsed_min']}, f)
log({'type': 'COMPLETE', 'elapsed_min': round((time.time() - t_all) / 60, 1)})
with open(PROG, 'w') as f:
    json.dump({'done': True, 'elapsed_min': round((time.time() - t_all) / 60, 1)}, f)
