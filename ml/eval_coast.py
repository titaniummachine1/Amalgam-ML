"""eval_coast.py — frozen-input (coast) baseline on rollout npz eval files.
Same physics + same demo flight flags as eval_tx.py, but inputs = none
every tick (no model). The floor every model must beat, same metric.
Usage: python ml/eval_coast.py <eval1.npz> [eval2.npz ...]
"""
import math
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from train_tx import HIST, FUT
from movesim import AIRCAP, rollout_free, yaw_to_fwd

HOR = [7, 13, 20]
R = 24.0


def main():
    hits = {h: [0, 0] for h in HOR}
    for path in sys.argv[1:]:
        d = np.load(path, allow_pickle=True)
        X, MSK, POS = d['X'], d['MSK'], d['POS']
        N = len(X)
        for gi in range(N):
            ey0 = math.degrees(math.atan2(X[gi, HIST - 1, 7], X[gi, HIST - 1, 8]))
            p0 = X[gi, HIST - 1, 0:3] * 1000.0
            v0 = X[gi, HIST - 1, 3:6] * 1000.0
            ms = X[gi, HIST - 1, 9] * 400.0
            inp = [(None, 0.0)] * FUT
            views = [yaw_to_fwd(ey0)] * FUT
            grf = [MSK[gi, k] != 2 for k in range(FUT)]
            pr = rollout_free(v0, p0, FUT, inp, grf, ms, AIRCAP, views)
            truth = POS[gi] * 100.0 + p0
            for h in HOR:
                e = float(np.linalg.norm(pr[h - 1] - truth[h - 1]))
                hits[h][1] += 1
                hits[h][0] += e < R
    for h in HOR:
        h1, h0 = hits[h]
        print('coast@%d: %.1f%% (%d/%d)' % (h, 100.0 * h1 / max(h0, 1), h1, h0), flush=True)


if __name__ == '__main__':
    main()
