"""Hit-rate @24hu on B-line eval (impulse-split), combat-masked subset available.
Compares coast vs denseC (gru_soldier_denseC.pt) vs dense66 (gru_soldier_dense66.pt)
on the SAME rows. Usage: setenv USEMASK=1 for combat-masked, else full evalB."""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
USEMASK = os.environ.get('USEMASK', '0') == '1'
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0
MAXV = 240.0


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


def rollout(net, X, fric):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = torch.from_numpy(X[b:b + 128])
            _, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * MAXV
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            hh = h
            hs = []
            for _ in range(66):
                a = net.head(hh[-1]) * MAXV
                if fric:
                    v = apply_friction(v, gr) + a * DT
                else:
                    v = v + a * DT
                p = p + v * DT
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / MAXV
                dyn = torch.cat([(v / MAXV).unsqueeze(1), sp.unsqueeze(1),
                                 gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1)], dim=-1)
                _, hh = net.gru(torch.cat([dyn, xb[:, -1:, 5:]], dim=-1), hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


d = np.load(os.path.join(D, 'evalB.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalBk.npz'))['K']
X, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
m = (k[:, 3] == 3)
if USEMASK:
    mk = np.load(os.path.join(D, 'evalCmask.npz'))['keep']
    m = m & mk
X, H = X[m], H[m]
idx = np.random.RandomState(1).choice(len(X), min(3000, len(X)), replace=False)
X, H = X[idx], H[idx]
print(('masked' if USEMASK else 'full'), 'evalB soldier n =', len(X), flush=True)
v0 = X[:, -1, :2] * MAXV
nets = {}
for name, ck, fr in [('dense66', 'gru_soldier_dense66.pt', False),
                     ('denseC', 'gru_soldier_denseC.pt', True)]:
    net = Net()
    net.load_state_dict(torch.load(os.path.join(D, ck), map_location='cpu'))
    net.eval()
    nets[name] = rollout(net, X, fr)
for label, arr in [('coast', None), ('dense66', nets['dense66']), ('denseC', nets['denseC'])]:
    line = []
    for hi, h_ in enumerate(HOR):
        tgt = H[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        err = (np.abs((v0 * (h_ * DT))[mm] - tgt[mm]).max(axis=1) if label == 'coast'
               else np.abs(arr[mm][:, h_ - 1] - tgt[mm]).max(axis=1))
        line.append(f'{h_}:{(err < HITR).mean() * 100:.1f}%')
    print(label, 'hit-rate@24hu', ' '.join(line), flush=True)
