"""eval_cont.py — champion-protocol eval for continuous models (train_cont.py).
Greedy AR: predict (wishdir, speed, jump) per tick, feed back, sim-integrate
with frozen eye (back-clamp view only) + demo flight flags (v1 scope), score
@7/13/20 R=24. Same truth, same physics as eval_tx.py — numbers comparable.
Usage: python ml/eval_cont.py <ckpt.pt> <eval1.npz> [...]
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, __import__('os').path.dirname(__file__))
import train_cont
from train_cont import ContAR, HIST, FUT
from movesim import AIRCAP, rollout_free, yaw_to_fwd

HOR = [7, 13, 20]
R = 24.0


def main():
    ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    if 'cfg' in ckpt:
        train_cont.D, train_cont.NL, train_cont.NH, train_cont.FF = ckpt['cfg'][:4]
    net = ContAR()
    net.load_state_dict(ckpt['net'])
    net.eval()
    torch.set_num_threads(4)
    hits = {h: [0, 0] for h in HOR}
    werr = wtot = 0.0
    for path in sys.argv[2:]:
        d = np.load(path, allow_pickle=True)
        X, MSK = d['X'], d['MSK']
        POS = d['POS'] * 100.0
        N = len(X)
        B = 64
        for s in range(0, N, B):
            sl = slice(s, min(s + B, N))
            xb = torch.tensor(X[sl], dtype=torch.float32)
            n = xb.size(0)
            fin = torch.zeros((n, FUT, 3), dtype=torch.float32)
            jin = torch.zeros((n, FUT), dtype=torch.float32)
            pw = np.zeros((n, FUT, 2))
            ps = np.zeros((n, FUT))
            with torch.no_grad():
                for k in range(FUT):
                    w, sp, jlog, _ = net(xb, fin, jin.long())
                    wk = w[:, k].numpy()
                    sk = sp[:, k].numpy()
                    nl = np.linalg.norm(wk, axis=1, keepdims=True)
                    wk = np.where(nl > 1e-6, wk / np.maximum(nl, 1e-6), 0.0)
                    pw[:, k] = wk
                    ps[:, k] = sk
                    if k + 1 < FUT:
                        fin[:, k + 1, :2] = torch.tensor(wk)
                        fin[:, k + 1, 2] = torch.tensor(sk)
            for i in range(n):
                gi = s + i
                w = np.vectorize({0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}.get)(MSK[gi])
                ey0 = math.degrees(math.atan2(X[gi, HIST - 1, 7], X[gi, HIST - 1, 8]))
                p0 = X[gi, HIST - 1, 0:3] * 1000.0
                v0 = X[gi, HIST - 1, 3:6] * 1000.0
                ms = X[gi, HIST - 1, 9] * 400.0
                inp, views, grf = [], [], []
                for k in range(FUT):
                    sp = float(ps[i, k]) * 400.0
                    if np.linalg.norm(pw[i, k]) > 1e-9 and sp > 25.0:
                        inp.append((pw[i, k] / max(np.linalg.norm(pw[i, k]), 1e-9), sp))
                    else:
                        inp.append((None, 0.0))
                    views.append(yaw_to_fwd(ey0))
                    grf.append(MSK[gi, k] != 2)
                pr = rollout_free(v0, p0, FUT, inp, grf, ms, AIRCAP, views)
                truth = POS[gi] + p0
                for h in HOR:
                    e = float(np.linalg.norm(pr[h - 1] - truth[h - 1]))
                    hits[h][1] += 1
                    hits[h][0] += e < R
    for h in HOR:
        h1, h0 = hits[h]
        print('hit@%d: %.1f%% (%d/%d)' % (h, 100.0 * h1 / max(h0, 1), h1, h0), flush=True)
    print('NOTE: continuous model; same physics/truth as eval_tx; demo flags v1.', flush=True)


if __name__ == '__main__':
    main()
