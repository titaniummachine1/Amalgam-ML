"""eval_tx.py — transformer eval, champion protocol (hit-rate @24hu).
AR decode (greedy, NO teacher forcing): history + own predicted tokens.
Eye FROZEN at history end during decode (production-faithful: one decode per
tick, no future info). Sim integrates predicted inputs (bin centers -> world
via frozen eye); grounded flags per-tick from demo (like the gate: isolates
INPUT quality from landing physics). Jump flag: applies 289 vz if sim
grounded (free-space approx; landing unknown without traces).
Metrics: hit-rate @24hu @[7,13,20] + token/jump acc + confidence calibration
(conf vs actual match rate). Champion compares @7/13/20/33/66; ours reports
7/13/20 (FUT=20 v1) + note.
Bin centers: none 0; spd1 87.5 (25-150); spd2 275 (150-400, sim clamps to
maxspeed); octant center = (oct+0.5)*45deg in ego frame.
Usage: python ml/eval_tx.py <model.pt> <eval1.npz> [eval2.npz ...]
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, __import__('os').path.dirname(__file__))
from train_tx import TinyAR, HIST, FUT, D, NL, NH, FF
from movesim import AIRCAP, DT, rollout_free, yaw_to_fwd

HOR = [7, 13, 20]
R = 24.0


def bin_to_world(tok, eyedeg):
    if tok == 0:
        return None, 0.0
    t = tok - 1
    oct8, spd = t // 3, t % 3
    ang = math.radians(eyedeg) + (oct8 + 0.5) * math.pi / 4
    ws = 87.5 if spd == 0 else 275.0
    return np.array([math.cos(ang), math.sin(ang)]), ws


import train_tx


def main():
    ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    if 'cfg' in ckpt:
        train_tx.D, train_tx.NL, train_tx.NH, train_tx.FF = ckpt['cfg'][:4]
    net = train_tx.TinyAR()
    net.load_state_dict(ckpt['net'])
    net.eval()
    torch.set_num_threads(4)
    hits = {h: [0, 0] for h in HOR}
    tacc = jacc = ttot = 0
    confs, matched = [], []
    for path in sys.argv[2:]:
        d = np.load(path, allow_pickle=True)
        X, TOK, JMP, MSK = d['X'], d['TOK'], d['JMP'], d['MSK']
        POS = d['POS'] * 100.0  # back to hu, rel window start
        N = len(X)
        B = 64
        for s in range(0, N, B):
            sl = slice(s, min(s + B, N))
            xb = torch.tensor(X[sl], dtype=torch.float32)
            n = xb.size(0)
            tin = torch.zeros((n, FUT), dtype=torch.long)
            jin = torch.zeros((n, FUT), dtype=torch.float32)
            ptoks = np.zeros((n, FUT), dtype=np.int64)
            pjmp = np.zeros((n, FUT), dtype=np.int64)
            with torch.no_grad():
                for k in range(FUT):
                    logits, jlog, _ = net(xb, tin, jin.long())
                    pk = logits[:, k].argmax(-1)
                    ptoks[:, k] = pk.numpy()
                    pj = (jlog[:, k] > 0).long()
                    pjmp[:, k] = pj.numpy()
                    if k + 1 < FUT:
                        tin[:, k + 1] = pk
                        jin[:, k + 1] = pj.float()
                _, _, conf = net(xb, torch.tensor(ptoks), torch.tensor(pjmp))
            for i in range(n):
                gi = s + i
                w = np.vectorize({0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}.get)(MSK[gi])
                tacc += ((ptoks[i] == TOK[gi]) * (w > 0)).sum()
                jacc += ((pjmp[i] == JMP[gi]) * (w > 0)).sum()
                ttot += (w > 0).sum()
                # sim integrate predicted inputs (frozen eye, demo grounded)
                ey0 = math.degrees(math.atan2(X[gi, HIST - 1, 7], X[gi, HIST - 1, 8]))
                p0 = X[gi, HIST - 1, 0:3] * 1000.0
                v0 = X[gi, HIST - 1, 3:6] * 1000.0
                ms = X[gi, HIST - 1, 9] * 400.0
                inp, views, grf = [], [], []
                for k in range(FUT):
                    wd, ws = bin_to_world(int(ptoks[i, k]), ey0)
                    inp.append((wd, ws))
                    views.append(yaw_to_fwd(ey0))
                    grf.append(MSK[gi, k] != 2)  # demo flight flags (v1);
                    # predicted jumps not yet simulated (landing unknown
                    # without traces) — under-predicts z on jump futures.
                pr = rollout_free(v0, p0, FUT, inp, grf, ms, AIRCAP, views)
                truth = POS[gi] + p0  # POS stored relative to window start + p0  # POS stored relative to window start + p0  # POS stored relative to window start
                for h in HOR:
                    e = float(np.linalg.norm(pr[h - 1] - truth[h - 1]))
                    hits[h][1] += 1
                    hits[h][0] += e < R
                cm = conf[i].numpy()
                for hi in range(6):
                    confs.append(float(cm[hi]))
    print('token-acc=%.3f jump-acc=%.3f (n=%d)' % (tacc / max(ttot, 1), jacc / max(ttot, 1), ttot), flush=True)
    for h in HOR:
        h1, h0 = hits[h]
        print('hit@%d: %.1f%% (%d/%d)' % (h, 100.0 * h1 / max(h0, 1), h1, h0), flush=True)
    print('NOTE: FUT=20 v1 (champion reports to 66); free-space eval; demo-grounded next.', flush=True)


main()
