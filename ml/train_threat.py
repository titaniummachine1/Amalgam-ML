"""Threat-input training: GRU(20,96) = 14 base + 6 threat feats per step.
Warm start from denseC: copy 14-col input weights, zero-init 6 threat cols
(new features start ignored = safe). Dense power-law loss + friction unroll.
Threat FROZEN at T during rollout (matches planned inference; extrapolation later).
SUFFIX env selects line (B). OUT: gru_threat_<suf>.pt"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
SUF = os.environ.get('DS_SUFFIX', 'B')
TAG = 'threat' + SUF
LOG = os.path.join(TMP, TAG + '.log.jsonl')
PROG = os.path.join(TMP, TAG + '_progress.json')
WARM = os.path.join(D, os.environ.get('WARM_CKPT', 'gru_soldier_denseC.pt'))
DT = 1 / 66.0
HB = 24.0
W = np.maximum((np.arange(1, 67) / 66.0) ** 1.3718, 0.02).astype(np.float32)
WT = torch.from_numpy(W)
MAXV1 = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)
EPOCHS = 4


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(20, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def load(tag):
    d = np.load(os.path.join(D, tag + SUF + '.npz'), mmap_mode='r')
    k = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    tw = np.load(os.path.join(D, tag + SUF + '_threatwin.npz'))
    Xb = np.array(d['X'])
    TW = np.array(tw['TW'])
    sol = k[:, 3] == 3
    assert (TW.shape[0] == sol.sum()), (TW.shape, sol.sum())
    X = np.concatenate([Xb[sol], TW], axis=2)
    Y = np.array(d['Y'])[sol]
    W8 = np.array(d['W'])[sol]
    return X, Y, W8


def apply_friction(v, gr):
    sp = (v ** 2).sum(-1, keepdim=True).sqrt()
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


def load_dense(tag):
    k = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    dn = np.load(os.path.join(D, tag + SUF + '_dense66.npz'))
    sol = np.where(k[:, 3] == 3)[0]
    return dn['Hd'], dn['Vd'], sol


Xtr, Ytr, Wtr = load('train')
Xev, Yev, Wev = load('eval')


def _dense(tag):
    kk = np.load(os.path.join(D, tag + SUF + 'k.npz'))['K']
    ddn = np.load(os.path.join(D, tag + SUF + '_dense66.npz'))
    _sol = kk[:, 3] == 3
    return ddn['Hd'], ddn['Vd']
HtrD, VtrD = _dense('train')
HevD, VevD = _dense('eval')
print('train', Xtr.shape, 'eval', Xev.shape, flush=True)


def unroll(net, xb):
    mv = (xb[:, -1:, 5:14] * torch.from_numpy(MAXV1)).sum(-1, keepdim=True)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mv.squeeze(1)
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:14]
    thr_frozen = xb[:, -1:, 14:20]
    hh = h
    traj = []
    for _ in range(66):
        a = net.head(hh[-1]) * mv.squeeze(1)
        v = apply_friction(v, gr) + a * DT
        p = p + v * DT
        sp = (v ** 2).sum(-1, keepdim=True).sqrt() / mv.squeeze(1)
        dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1),
                         thr_frozen], dim=-1)
        _, hh = net.gru(torch.cat([dyn, static], dim=-1), hh)
        traj.append(p)
    return torch.stack(traj, dim=1)


def dense_loss(net, xb, Hb, Vb):
    traj = unroll(net, xb)
    tgt = torch.from_numpy(np.array(Hb))
    m = torch.from_numpy(np.array(Vb)).unsqueeze(-1).float()
    err = torch.abs(traj - tgt).amax(dim=-1, keepdim=True)
    inside = (err <= HB).float()
    lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
    lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
    per = ((lin + lout) * WT.view(1, -1, 1) * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
    return per.mean()


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


net = Net()
sd = torch.load(WARM, map_location='cpu')
with torch.no_grad():
    w_ih = torch.zeros(3 * 96, 20)
    w_ih[:, :14] = sd['gru.weight_ih_l0']
    sd['gru.weight_ih_l0'] = w_ih
    missing, unexpected = [], []
    try:
        net.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print('LOAD FAIL:', str(e)[:500], flush=True)
        raise
    assert net.gru.weight_ih_l0.shape == (288, 20)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
B = 128
N = len(Xtr)
best = 1e18
OUT = os.path.join(D, 'gru_' + TAG + '.pt')
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
        loss = dense_loss(net, xb, HtrD[idx], VtrD[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        bl.append(loss.item())
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        for i in range(0, len(Xev), 512):
            xb = torch.from_numpy(np.array(Xev[i:i + 512]))
            el.append(dense_loss(net, xb, HevD[i:i + 512], VevD[i:i + 512]).item())
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
