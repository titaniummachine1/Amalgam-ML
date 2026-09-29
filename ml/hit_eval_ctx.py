"""Hit-rate @24hu: coast vs v4 (5-dim) vs raw-ctx vs res-ctx (37-dim).
res: adds heur back each rollout step (same as its finetune)."""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0


class Net5(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(5, 64, batch_first=True)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class Net37(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(37, 80, batch_first=True)
        self.head = nn.Linear(80, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


d = np.load(os.path.join(D, 'eval3.npz'), mmap_mode='r')
e = np.load(os.path.join(D, 'eval_ctx.npz'), mmap_mode='r')
Xb, Y, H = np.array(d['X']), np.array(d['Y']), np.array(d['H'])
Es = np.repeat(np.array(e['E'])[:, None, :], 66, axis=1)
Xc = np.concatenate([Xb, Es], axis=2)
m = (np.abs(Y).max(axis=1) <= 40)
Xb, Xc, Y, H = Xb[m], Xc[m], Y[m], H[m]
print('n =', len(Xb), flush=True)

net4 = Net5()
net4.load_state_dict(torch.load(os.path.join(D, 'gru64_v4.pt'), map_location='cpu'))
net4.eval()
netraw = Net37()
netraw.load_state_dict(torch.load(os.path.join(D, 'gru_ctx_raw_v5.pt'), map_location='cpu'))
netraw.eval()
netres = Net37()
netres.load_state_dict(torch.load(os.path.join(D, 'gru_ctx_res_v5.pt'), map_location='cpu'))
netres.eval()

idx = np.random.RandomState(1).choice(len(Xb), min(2000, len(Xb)), replace=False)
Xb, Xc, H = Xb[idx], Xc[idx], H[idx]
v0 = Xb[:, -1, :2] * 240.0


def rollout(net, X, res=False):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = torch.from_numpy(X[b:b + 128])
            _, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * 240.0
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            static = xb[:, -1:, 5:] if xb.shape[2] > 5 else None
            heur = xb[:, -1, 32:34] * 240.0 if res else 0.0
            hh = h
            hs = []
            for _ in range(66):
                a = net.head(hh[-1]) * 240.0 + heur
                v = v + a * DT
                p = p + v * DT
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / 240.0
                dyn = torch.cat([v / 240.0, sp, gr, torch.zeros_like(sp)], dim=-1).unsqueeze(1)
                xin = torch.cat([dyn, static], dim=-1) if static is not None else dyn
                _, hh = net.gru(xin, hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


results = {}
coast = {hi: v0 * (h_ * DT) for hi, h_ in enumerate(HOR)}
print('rolling v4...', flush=True)
tr4 = rollout(net4, Xb)
print('rolling raw-ctx...', flush=True)
trraw = rollout(netraw, Xc)
print('rolling res-ctx...', flush=True)
trres = rollout(netres, Xc, res=True)
names = ['100', '200', '300', '500', '1000']
for label, arr in [('coast', None), ('v4', tr4), ('rawctx', trraw), ('resctx', trres)]:
    line = []
    for hi, h_ in enumerate(HOR):
        tgt = H[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        err = (np.abs(coast[hi][mm] - tgt[mm]).max(axis=1) if label == 'coast'
               else np.abs(arr[mm][:, h_ - 1] - tgt[mm]).max(axis=1))
        line.append(f'{names[hi]}:{(err < HITR).mean() * 100:.1f}%')
    print(label, 'hit-rate@24hu', ' '.join(line), flush=True)
