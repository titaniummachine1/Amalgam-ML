"""Quick standing eval of gru_all_fp32.pt: weighted single-step MSE + small rollout hit-rate."""
import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
ev = np.load(os.path.join(D, 'evalA.npz'), mmap_mode='r')
Xev, Yev, Wev, Hev = ev['X'], ev['Y'], ev['W'], ev['H']
m = (np.abs(Yev).max(axis=1) <= 40)
Xev, Yev, Wev, Hev = Xev[m], Yev[m], Wev[m], Hev[m]
print('eval', Xev.shape, flush=True)
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0


class Net(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(14, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


net = Net()
net.load_state_dict(torch.load(os.path.join(D, 'gru_all_fp32.pt'), map_location='cpu'))
net.eval()
with torch.no_grad():
    tot = 0.0
    nb = 0
    for i in range(0, len(Xev), 2048):
        xb = torch.from_numpy(np.array(Xev[i:i + 2048]))
        yb = torch.from_numpy(np.array(Yev[i:i + 2048]))
        wb = torch.from_numpy(np.array(Wev[i:i + 2048])).view(-1, 1)
        tot += ((((net(xb) - yb) ** 2) * wb).mean() * len(xb)).item()
        nb += len(xb)
    print('wMSE eval:', tot / nb, flush=True)

idx = np.random.RandomState(1).choice(len(Xev), min(1500, len(Xev)), replace=False)
Xs, Hs = Xev[idx], Hev[idx]
v0 = Xs[:, -1, :2]
# NOTE: X normalized per-class; de-norm needs per-class MAXV from one-hot (dims 5:14)
oh = Xs[:, -1, 5:14]
maxv = (oh * np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)).sum(axis=1, keepdims=True)
v0hu = v0 * maxv
outs = []
with torch.no_grad():
    for b in range(0, len(Xs), 128):
        xb = torch.from_numpy(Xs[b:b + 128])
        mv = torch.from_numpy(maxv[b:b + 128])
        _, h = net.gru(xb)
        v = xb[:, -1, :2].clone() * mv
        p = torch.zeros_like(v)
        gr = xb[:, -1, 3:4]
        hh = h
        hs = []
        for _ in range(66):
            a = net.head(hh[-1]) * mv
            v = v + a * DT
            p = p + v * DT
            sp = (v ** 2).sum(-1, keepdim=True).sqrt() / mv
            xin = torch.cat([(v / mv).unsqueeze(1), sp.unsqueeze(1), gr.unsqueeze(1),
                               torch.zeros_like(sp).unsqueeze(1), xb[:, -1:, 5:]], dim=-1)
            _, hh = net.gru(xin, hh)
            hs.append(p.clone())
        outs.append(torch.stack(hs, dim=1).numpy())
tr = np.concatenate(outs, axis=0)
for label, arr in [('coast', None), ('allfp32', tr)]:
    line = []
    for hi, h_ in enumerate(HOR):
        tgt = Hs[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        err = (np.abs((v0hu * (h_ * DT))[mm] - tgt[mm]).max(axis=1) if label == 'coast'
               else np.abs(arr[mm][:, h_ - 1] - tgt[mm]).max(axis=1))
        line.append(f'{h_}:{(err < HITR).mean() * 100:.1f}%')
    print(label, 'hit-rate@24hu', ' '.join(line), flush=True)
