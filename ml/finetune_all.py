"""All-class hitbox-loss finetune. GRU(14,96) from gru_all_fp32.pt.
Per-class de-norm via one-hot (dims 5:14). Logs per-epoch mean/std, writes
progress JSON for status queries. -> gru_all_v5.pt"""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
LOG = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode\finetune_all.jsonl'
PROG = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode\finetune_all_progress.json'
MAXV = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)
tr = np.load(os.path.join(D, 'trainA.npz'), mmap_mode='r')
ev = np.load(os.path.join(D, 'evalA.npz'), mmap_mode='r')
Xtr, Ytr, Wtr, Htr = tr['X'], tr['Y'], tr['W'], tr['H']
Xev, Yev, Wev, Hev = ev['X'], ev['Y'], ev['W'], ev['H']
mtr = (np.abs(Ytr).max(axis=1) <= 40)
mev = (np.abs(Yev).max(axis=1) <= 40)
Xtr, Ytr, Wtr, Htr = Xtr[mtr], Ytr[mtr], Wtr[mtr], Htr[mtr]
Xev, Yev, Wev, Hev = Xev[mev], Yev[mev], Wev[mev], Hev[mev]
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HB = 24.0


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(14, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def maxv_of(xb):
    return (xb[:, -1:, 5:] * torch.from_numpy(MAXV)).sum(-1, keepdim=True)


def unroll(net, xb):
    mv = maxv_of(xb)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mv.squeeze(1)
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:]
    hh = h
    traj = []
    for _ in range(66):
        a = net.head(hh[-1]) * mv.squeeze(1)
        v = v + a * DT
        p = p + v * DT
        sp = (v ** 2).sum(-1, keepdim=True).sqrt() / mv.squeeze(1)
        dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static], dim=-1), hh)
        traj.append(p)
    return torch.stack(traj, dim=1), net.head(hh[-1]) * mv.squeeze(1)


def hitbox_loss(net, xb, yb, wb, Hb):
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
    return l + anchor


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


net = Net()
net.load_state_dict(torch.load(os.path.join(D, 'gru_all_fp32.pt'), map_location='cpu'))
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
B = 128
N = len(Xtr)
best = 1e18
OUT = os.path.join(D, 'gru_all_v5.pt')
t_start = time.time()
# fixed val-probe subset (single-step, fast): how-far-from-perfect signal mid-epoch
probe_idx = np.random.RandomState(0).choice(len(Xev), 2048, replace=False)
Xpb = torch.from_numpy(np.array(Xev[probe_idx]))
Ypb = torch.from_numpy(np.array(Yev[probe_idx]))
Wpb = torch.from_numpy(np.array(Wev[probe_idx])).view(-1, 1)
LOG_EVERY = 200
for ep in range(4):
    net.train()
    perm = np.random.permutation(N)
    batch_losses = []
    t0 = time.time()
    for bi, i in enumerate(range(0, N, B)):
        idx = perm[i:i + B]
        xb = torch.from_numpy(np.array(Xtr[idx]))
        yb = torch.from_numpy(np.array(Ytr[idx]))
        wb = torch.from_numpy(np.array(Wtr[idx])).view(-1, 1)
        Hb = Htr[idx]
        opt.zero_grad()
        l = hitbox_loss(net, xb, yb, wb, Hb)
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        batch_losses.append(l.item())
        if (bi + 1) % LOG_EVERY == 0:
            bl = np.array(batch_losses)
            with torch.no_grad():
                was_training = net.training
                net.eval()
                probe = float((((net(Xpb) - Ypb) ** 2) * Wpb).mean())
                if was_training:
                    net.train()
            log({'type': 'batch', 'ep': ep, 'batch': bi + 1, 'of_batches': N // B,
                 'train_mean': float(bl.mean()), 'train_std': float(bl.std()),
                 'valprobe_singlestep': probe,
                 'elapsed_min': round((time.time() - t_start) / 60, 1)})
            with open(PROG, 'w') as f:
                json.dump({'epoch': ep, 'of': 4, 'batch': bi + 1, 'best': best,
                           'train_mean': float(bl.mean()), 'train_std': float(bl.std()),
                           'valprobe': probe, 'done': False, 'ckpt': OUT}, f)
    bl = np.array(batch_losses)
    net.eval()
    with torch.no_grad():
        evl = []
        evw = []
        for i in range(0, len(Xev), 512):
            xb = torch.from_numpy(np.array(Xev[i:i + 512]))
            yb = torch.from_numpy(np.array(Yev[i:i + 512]))
            wb = torch.from_numpy(np.array(Wev[i:i + 512])).view(-1, 1)
            Hb = Hev[i:i + 512]
            evl.append(hitbox_loss(net, xb, yb, wb, Hb).item())
            evw.append(len(xb))
        ev_arr = np.array(evl)
        ev_mean = float(np.average(ev_arr, weights=evw))
        ev_std = float(np.sqrt(np.average((ev_arr - ev_mean) ** 2, weights=evw)))
    rec = {'ep': ep, 'train_mean': float(bl.mean()), 'train_std': float(bl.std()),
           'eval_mean': ev_mean, 'eval_std': ev_std,
           'ep_sec': round(time.time() - t0, 1),
           'elapsed_min': round((time.time() - t_start) / 60, 1)}
    improved = ev_mean < best
    if improved:
        best = ev_mean
        torch.save(net.state_dict(), OUT)
    rec['best'] = best
    rec['saved'] = improved
    log(rec)
    with open(PROG, 'w') as f:
        json.dump({'epoch': ep, 'of': 4, 'best': best, 'last': rec,
                   'done': ep == 3, 'ckpt': OUT}, f)
print('COMPLETE best', best, flush=True)
