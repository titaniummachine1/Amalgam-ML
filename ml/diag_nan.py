import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_both27 as T

torch.set_num_threads(4)
B = 128
# find a skipping batch: scan blocks' first batch
net = T.Net27()
net.load_state_dict(torch.load(T.OUT, map_location='cpu'))
net.train()
found = None
for a in range(0, T.N_TR, 2048 * 8):
    bb = min(a + 128, T.N_TR)
    xb = torch.from_numpy(np.concatenate([np.array(T.Xtr[a:bb]), np.array(T.Wtr[a:bb])], axis=2))
    sb = torch.from_numpy(np.array(T.S0tr[a:bb]))
    with torch.no_grad():
        l = T.dense_loss(net, xb, sb, np.array(T.Htr[a:bb]), np.array(T.Vtr[a:bb]),
                         Jm=np.array(T.JHtr[a:bb]), rows=np.arange(a, bb))
    if not np.isfinite(l.item()):
        found = (a, bb)
        break
print('first bad block at rows', found, flush=True)
if found is None:
    print('no NaN in scan?!')
    sys.exit(0)
a, bb = found
xb = torch.from_numpy(np.concatenate([np.array(T.Xtr[a:bb]), np.array(T.Wtr[a:bb])], axis=2))
sb = torch.from_numpy(np.array(T.S0tr[a:bb]))
Hb = np.array(T.Htr[a:bb])
Vb = np.array(T.Vtr[a:bb])
Jb = np.array(T.JHtr[a:bb])
print('xb finite:', torch.isfinite(xb).all().item(), 'sb:', torch.isfinite(sb).all().item())
print('Hb has nan:', np.isnan(Hb).sum(), 'has inf:', np.isinf(Hb).sum())
print('Vb sum:', Vb.sum(), 'Jb sum:', Jb.sum())
with torch.no_grad():
    traj, acc = T.unroll(net, xb, sb, nsteps=T.HOR, ret_acc=True)
print('traj finite:', torch.isfinite(traj).all().item(), 'acc finite:', torch.isfinite(acc).all().item())
# replicate dense_loss pieces
tgt = torch.nan_to_num(torch.from_numpy(Hb), nan=0.0)
m = torch.from_numpy(Vb).unsqueeze(-1).float() * torch.from_numpy(Jb).unsqueeze(-1).float()
err = torch.abs(traj - tgt).amax(dim=-1, keepdim=True)
print('err finite:', torch.isfinite(err).all().item(), 'err max:', err.max().item())
inside = (err <= T.HB).float()
lin = torch.log((err + 1.0) / (T.HB + 1.0)) * inside
lout = 4.0 * ((err - T.HB) / T.HB).clamp_min(0.0) ** 2 * (1.0 - inside)
print('lin finite:', torch.isfinite(lin).all().item(), 'lout finite:', torch.isfinite(lout).all().item())
base = lin + lout
mv = (xb[:, -1:, 5:14] * torch.from_numpy(T.MAXV1)).sum(-1, keepdim=True)
v0 = xb[:, -1, :2] * mv.squeeze(1)
gr = xb[:, -1, 3:4]
Hd0 = torch.cat([torch.zeros_like(tgt[:, :1]), tgt], dim=1)
Vt = (Hd0[:, 1:] - Hd0[:, :-1]) / T.DT
Vf = T.apply_friction(torch.cat([v0.unsqueeze(1), Vt[:, :-1]], dim=1), gr.unsqueeze(1))
Aneed = (Vt - Vf) / T.DT
print('Aneed finite:', torch.isfinite(Aneed).all().item(), 'Aneed max:', Aneed.abs().max().item())
bl = (acc.detach() - Aneed).abs().mean(-1, keepdim=True)
print('bl finite:', torch.isfinite(bl).all().item(), 'bl max:', bl.max().item())
tick = ((base * m).sum(0) / m.sum(0).clamp_min(1)).squeeze(-1).detach()
print('tick finite:', torch.isfinite(tick).all().item(), 'tick:', tick[:5].numpy(), '...', tick[-3:].numpy())
print('m.sum(0) zeros:', (m.sum(0).squeeze(-1) == 0).sum().item())
