import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_both27 as T

torch.set_num_threads(4)
xb = torch.from_numpy(np.concatenate([np.array(T.Xtr[:128]), np.array(T.Wtr[:128])], axis=2))
sb = torch.from_numpy(np.array(T.S0tr[:128]))
Hb = torch.from_numpy(np.array(T.Htr[:128]))
tgt = torch.nan_to_num(Hb, nan=0.0)
mv = (xb[:, -1:, 5:14] * torch.from_numpy(T.MAXV1)).sum(-1, keepdim=True)
print('xb', tuple(xb.shape), 'mv', tuple(mv.shape))
gr = xb[:, -1, 3:4]
print('gr', tuple(gr.shape))
cap = torch.where(gr > 0.5, 14.0 * mv * (4000.0 / 3360.0), torch.full_like(mv, 3000.0))
print('cap', tuple(cap.shape))
Hd0 = torch.cat([torch.zeros_like(tgt[:, :1]), tgt], dim=1)
Vt = (Hd0[:, 1:] - Hd0[:, :-1]) / T.DT
print('Vt', tuple(Vt.shape))
v0 = xb[:, -1, :2] * mv.squeeze(1)
Vf = T.apply_friction(torch.cat([v0.unsqueeze(1), Vt[:, :-1]], dim=1), gr.unsqueeze(1))
Aneed = (Vt - Vf) / T.DT
print('Aneed', tuple(Aneed.shape))
print('amax', tuple(Aneed.abs().amax(dim=-1, keepdim=True).shape))
