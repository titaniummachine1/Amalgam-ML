"""Gap scanner: finds MODEL BLIND SPOTS (not noise). Runs the current policy over
an eval subset, computes per-tick blame, saves rows where blame is HIGH but
aneed is PHYSICAL (<3360 ground-physics cap): the model is wrong about real,
learnable behavior. Output TMP/gap_rows.jsonl {row(eval soldier idx),h,blame,
aneed,gr,sp}. Usage: python ml/scan_gaps.py [n=2500]"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_both27 as T

torch.set_num_threads(4)
D = T.D
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500

P = os.path.join(D, 'evalB_X20sol')
nn_ = int(os.path.getsize(P + '.raw') // (66 * 20 * 4))
X = np.memmap(P + '.raw', dtype=np.float32, mode='r', shape=(nn_, 66, 20))
W7 = np.array(np.load(os.path.join(D, 'evalB_w7.npz'))['W7'])
H = np.memmap(P.replace('X20sol', 'HD132') + '.raw', dtype=np.float32, mode='r',
              shape=(nn_, 132, 2))
V = np.load(P.replace('X20sol', 'VD132') + '.npy', mmap_mode='r')
tw = np.load(os.path.join(D, 'evalB_threatwin.npz'))
S0 = np.array(tw['TS'])[:, -1, :]
tw.close()
idx = np.random.RandomState(7).choice(nn_, min(N, nn_), replace=False)
print('scanning %d rows' % len(idx), flush=True)

net = T.Net27()
net.load_state_dict(torch.load(T.OUT, map_location='cpu'))
net.eval()
out = open(os.path.join(T.TMP, 'gap_rows.jsonl'), 'w')
n_gap = n_noise = 0
with torch.no_grad():
    for b in range(0, len(idx), 128):
        jb = idx[b:b + 128]
        xb = torch.from_numpy(np.concatenate([np.array(X[jb]), W7[jb]], axis=2))
        sb = torch.from_numpy(np.array(S0[jb]))
        Hb = torch.from_numpy(np.array(H[jb]))
        Vb = torch.from_numpy(np.array(V[jb])).unsqueeze(-1).float()
        traj, acc = T.unroll(net, xb, sb, nsteps=T.HOR, ret_acc=True)
        tgt = torch.nan_to_num(Hb, nan=0.0)
        mv = (xb[:, -1:, 5:14] * torch.from_numpy(T.MAXV1)).sum(-1, keepdim=True)
        v0 = xb[:, -1, :2] * mv.squeeze(1)
        gr = xb[:, -1, 3:4]
        Hd0 = torch.cat([torch.zeros_like(tgt[:, :1]), tgt], dim=1)
        Vt = (Hd0[:, 1:] - Hd0[:, :-1]) / T.DT
        Vf = T.apply_friction(torch.cat([v0.unsqueeze(1), Vt[:, :-1]], dim=1),
                              gr.unsqueeze(1))
        Aneed = (Vt - Vf) / T.DT
        bl = (acc - Aneed).abs().mean(-1)
        an = Aneed.abs().amax(-1)
        Vm = Vb.squeeze(-1).bool()
        Vm2 = (Vm & torch.cat([Vm[:, :1], Vm[:, :-1]], dim=1)).numpy()
        bln, ann = bl.numpy(), an.numpy()
        for r in range(len(jb)):
            for h in range(T.HOR):
                if not Vm2[r, h]:
                    continue
                if bln[r, h] > 500 and ann[r, h] < 3360:
                    out.write(json.dumps({'row': int(jb[r]), 'h': int(h),
                                          'blame': round(float(bln[r, h]), 1),
                                          'aneed': round(float(ann[r, h]), 1),
                                          'gr': float(gr[r].item()),
                                          'sp': round(float(
                                              np.linalg.norm(xb[r, -1, :2].numpy())
                                              * 240.0), 0)}) + '\n')
                    n_gap += 1
                elif ann[r, h] >= 3360:
                    n_noise += 1
        print('batch %d/%d gaps=%d noise-ticks=%d' % (b, len(idx), n_gap, n_noise), flush=True)
out.close()
print('GAP rows saved: %d (noise ticks seen: %d)' % (n_gap, n_noise), flush=True)
