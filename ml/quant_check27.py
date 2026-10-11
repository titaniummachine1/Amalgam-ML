"""Quant gate for both27: fp32 ckpt vs DEQUANTIZED SHIPPED HEADER bytes.
Reads Amalgam/.../weights_both27.h (int8 + per-row scales), rebuilds the net,
rolls out both on a 1000-row eval subset with the exact training dynamics,
reports hit-rate @24hu deltas. Ship iff <=0.5pp everywhere.
Usage: python ml/quant_check27.py"""
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_both27 import slen, apply_friction, MAXV1

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
HDIR = r'C:\gitProjects\amalgam\Amalgam\src\Features\Simulation\BehaviorPrediction'
HOR = [7, 13, 20, 33, 66]
DT = 1 / 66.0
HITR = 24.0


class Net27(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(27, 96, batch_first=True)
        self.head = nn.Linear(96, 2)


def load_header():
    src = open(os.path.join(HDIR, 'weights_both27.h')).read()
    out = {}

    def arr(name):
        m = re.search(r'kW_%s\[(\d+)\] = \{([^}]*)\}' % name, src)
        return np.fromstring(m.group(2), sep=',', dtype=np.int8)

    def scl(name):
        m = re.search(r'kScale_%s = \{([^}]*)\}' % name, src)
        if m:
            return np.fromstring(m.group(1).replace('f', ''), sep=',', dtype=np.float32)
        m = re.search(r'kScale_%s = ([0-9.e+-]+)f;' % name, src)
        return np.float32(m.group(1))

    names = ['gru_weight_ih_l0', 'gru_weight_hh_l0', 'gru_bias_ih_l0',
             'gru_bias_hh_l0', 'head_weight', 'head_bias']
    shapes = {'gru_weight_ih_l0': (288, 27), 'gru_weight_hh_l0': (288, 96),
              'gru_bias_ih_l0': (288,), 'gru_bias_hh_l0': (288,),
              'head_weight': (2, 96), 'head_bias': (2,)}
    for n in names:
        q, s = arr(n), scl(n)
        if isinstance(s, np.ndarray):
            w = q.astype(np.float32).reshape(shapes[n]) * s[:, None]
        else:
            w = q.astype(np.float32).reshape(shapes[n]) * float(s)
        out[n.replace('_l0', '.l0').replace('gru_weight', 'gru.weight').replace(
            'gru_bias', 'gru.bias').replace('head_weight', 'head.weight').replace(
            'head_bias', 'head.bias')] = torch.from_numpy(w)
    return out


fp = Net27()
fp.load_state_dict(torch.load(os.path.join(D, 'gru_both27_best.pt'), map_location='cpu'))
fp.eval()
qd = load_header()
qu = Net27()
qu.load_state_dict(qd)
qu.eval()
print('header parse ok', flush=True)

d = np.load(os.path.join(D, 'evalB.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalBk.npz'))['K']
tw = np.load(os.path.join(D, 'evalB_threatwin.npz'), mmap_mode='r')
w7 = np.load(os.path.join(D, 'evalB_w7.npz'), mmap_mode='r')['W7']
Xb = np.array(d['X'])
TW = np.array(tw['TW'])
TS = np.array(tw['TS'])[:, -1, :]
H = np.array(d['H'])
Y = np.array(d['Y'])
sol = np.where(k[:, 3] == 3)[0]
Xb, TW, TS = Xb[sol], TW, TS
H, Y = H, Y[sol]
X27 = np.concatenate([Xb, TW, W7], axis=2)
m = (np.abs(Y).max(axis=1) <= 40)
X27, TS, H = X27[m], TS[m], H[m]
idx = np.random.RandomState(1).choice(len(X27), min(1000, len(X27)), replace=False)
X27, TS, H = X27[idx], TS[idx], H[idx]
v0 = X27[:, -1, :2] * 240.0


def rollout(net, X, S):
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = torch.from_numpy(X[b:b + 128])
            sb = torch.from_numpy(S[b:b + 128])
            mv = (xb[:, -1:, 5:14] * torch.from_numpy(MAXV1)).sum(-1, keepdim=True)
            _, h = net.gru(xb)
            v = xb[:, -1, :2].clone() * mv.squeeze(1)
            p = torch.zeros_like(v)
            gr = xb[:, -1, 3:4]
            static = xb[:, -1:, 5:14]
            wstat = xb[:, -1:, 20:27]
            q = sb[:, 0:2].clone()
            vth = sb[:, 2:4].clone()
            present = sb[:, 4:5]
            thr_frozen = xb[:, -1:, 14:20]
            nspd = torch.clamp(slen(vth) / 3000.0, 0.0, 1.0)
            cnt = thr_frozen[:, 0, 3:4]
            mem = thr_frozen[:, 0, 4:5]
            hh = h
            hs = []
            for _ in range(66):
                a = net.head(hh[-1]) * mv.squeeze(1)
                v = apply_friction(v, gr) + a * DT
                p = p + v * DT
                q = q + vth * DT
                rel = q - p
                dist = slen(rel, dim=1).clamp_min(1e-6)
                ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
                ang = torch.atan2(rel[:, 1:2], rel[:, 0:1])
                infov = (ang.abs() < (np.pi / 4)).float()
                appr = -((vth - v) * rel).sum(-1, keepdim=True) / dist
                tti = torch.where(appr > 50.0, torch.clamp(dist / appr.clamp_min(1e-9) / 5.0, 0.0, 1.0),
                                  torch.ones_like(dist))
                live = torch.cat([ndist, nspd, infov, cnt, mem, tti], dim=-1).unsqueeze(1)
                thr = present.unsqueeze(1) * live + (1.0 - present.unsqueeze(1)) * thr_frozen
                sp = slen(v) / mv.squeeze(1)
                dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                                 gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1),
                                 thr], dim=-1)
                _, hh = net.gru(torch.cat([dyn, static, wstat], dim=-1), hh)
                hs.append(p.clone())
            outs.append(torch.stack(hs, dim=1).numpy())
    return np.concatenate(outs, axis=0)


print('rolling fp32...', flush=True)
trF = rollout(fp, X27, TS)
print('rolling int8-header...', flush=True)
trQ = rollout(qu, X27, TS)
print('--- fp32 vs int8-header ---', flush=True)
ok = True
for hi, h_ in enumerate(HOR):
    tgt = H[:, hi]
    mm = ~np.isnan(tgt[:, 0])
    ef = np.abs(trF[mm][:, h_ - 1] - tgt[mm]).max(axis=1)
    eq = np.abs(trQ[mm][:, h_ - 1] - tgt[mm]).max(axis=1)
    df = (ef < HITR).mean() * 100
    dq = (eq < HITR).mean() * 100
    flag = '' if abs(df - dq) <= 0.5 else '  <-- EXCEEDS 0.5pp, DO NOT SHIP'
    if abs(df - dq) > 0.5:
        ok = False
    print('  h=%d fp32=%.1f%% int8=%.1f%% (d=%+.2fpp)%s' % (h_, df, dq, dq - df, flag), flush=True)
print('GATE:', 'PASS' if ok else 'FAIL', flush=True)
