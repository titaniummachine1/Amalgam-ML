"""Quant check: FP16 vs INT8 weight quantization of champion v4.
Measures rollout hit-rate @24hu on eval subset + per-step timing.
INT8: per-tensor symmetric for GRU weights, per-column for head; activations fp32."""
import os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
ev = np.load(os.path.join(D, 'eval3.npz'), mmap_mode='r')
Xev, Yev, Hev = ev['X'], ev['Y'], ev['H']
m = (np.abs(Yev).max(axis=1) <= 40)
Xev, Hev = Xev[m], Hev[m]
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(5, 64, batch_first=True)
        self.head = nn.Linear(64, 2)


def quantize_int8(net):
    q = Net()
    q.load_state_dict(net.state_dict())
    with torch.no_grad():
        for name, p in q.named_parameters():
            if 'weight' in name:
                s = p.abs().max() / 127.0
                p.copy_((torch.round(p / s).clamp(-128, 127) * s))
    return q


def to_fp16(net):
    q = Net()
    q.load_state_dict(net.state_dict())
    return q.half()


fp32 = Net()
fp32.load_state_dict(torch.load(os.path.join(D, 'gru64_v4.pt'), map_location='cpu'))
fp32.eval()
fp16 = to_fp16(fp32).eval()
int8 = quantize_int8(fp32).eval()

idx = np.random.RandomState(1).choice(len(Xev), min(1500, len(Xev)), replace=False)
Xs, Hs = Xev[idx], Hev[idx]
v0 = Xs[:, -1, :2] * 240.0


def rollout(net, X, half=False):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = torch.from_numpy(X[b:b + 128])
            if half:
                xb = xb.half()
            _, h = net.gru(xb)
            v = (xb[:, -1, :2].float() * 240.0)
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4].float()
            hh = h
            hs = []
            for _ in range(66):
                a = net.head(hh[-1]).float() * 240.0
                v = v + a * DT
                p = p + v * DT
                sp = (v ** 2).sum(-1, keepdim=True).sqrt() / 240.0
                xin = torch.cat([v / 240.0, sp, gr, torch.zeros_like(sp)], dim=-1).unsqueeze(1)
                if half:
                    xin = xin.half()
                _, hh = net.gru(xin, hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


for label, net, half in [('fp32', fp32, False), ('fp16', fp16, True), ('int8', int8, False)]:
    t = time.perf_counter()
    tr = rollout(net, Xs, half)
    dt = time.perf_counter() - t
    line = []
    for hi, h_ in enumerate(HOR):
        tgt = Hs[:, hi]
        mm = ~np.isnan(tgt[:, 0])
        err = np.abs(tr[mm][:, h_ - 1] - tgt[mm]).max(axis=1)
        line.append(f'{h_}:{(err < HITR).mean() * 100:.1f}%')
    print(f'{label}: hit-rate', ' '.join(line), f'| {dt / len(Xs) * 1000:.1f}ms/rollout(torch)', flush=True)
