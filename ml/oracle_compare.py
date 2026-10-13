"""oracle_compare.py — quantization ceiling proof (no training).
On eval npz rows: (a) binned oracle (TOK bins -> world inputs, frozen eye,
demo flags) vs (b) continuous oracle (WC wishdir/speed, frozen eye, demo
flags). Same sim, same truth. If (b) >> (a) and (b) > coast, regression head
is THE fix. If (b) <= coast, eye prediction (or more) is needed too.
Usage: python ml/oracle_compare.py <eval_all.txt>
"""
import math
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from train_tx import HIST, FUT
from eval_tx import bin_to_world
from movesim import AIRCAP, rollout_free, yaw_to_fwd

HOR = [7, 13, 20]
R = 24.0


def roll(wish, v0, p0, ms, views, grf):
    return rollout_free(v0, p0, FUT, wish, grf, ms, AIRCAP, views)


def main():
    files = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    hb = {h: [0, 0] for h in HOR}
    hc = {h: [0, 0] for h in HOR}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        X, TOK, MSK, POS = d['X'], d['TOK'], d['MSK'], d['POS']
        wc = np.load(fp + '.cont.npz', allow_pickle=True)['WC']
        N = len(X)
        for gi in range(N):
            ey0 = math.degrees(math.atan2(X[gi, HIST - 1, 7], X[gi, HIST - 1, 8]))
            p0 = X[gi, HIST - 1, 0:3] * 1000.0
            v0 = X[gi, HIST - 1, 3:6] * 1000.0
            ms = X[gi, HIST - 1, 9] * 400.0
            views = [yaw_to_fwd(ey0)] * FUT
            grf = [MSK[gi, k] != 2 for k in range(FUT)]
            # (a) binned oracle
            inp_b = []
            for k in range(FUT):
                wd, ws = bin_to_world(int(TOK[gi, k]), ey0)
                inp_b.append((wd, ws))
            # (b) continuous oracle (wish world-frame? WC stored world wishdir
            # from inverse_*; speed denormalized)
            inp_c = []
            for k in range(FUT):
                w = wc[gi, k]
                if abs(w[0]) + abs(w[1]) > 1e-9:
                    inp_c.append((np.array([w[0], w[1]]), float(w[2]) * 400.0))
                else:
                    inp_c.append((None, 0.0))
            pr_b = roll(inp_b, v0, p0, ms, views, grf)
            pr_c = roll(inp_c, v0, p0, ms, views, grf)
            truth = POS[gi] * 100.0 + p0
            for h in HOR:
                eb = float(np.linalg.norm(pr_b[h - 1] - truth[h - 1]))
                ec = float(np.linalg.norm(pr_c[h - 1] - truth[h - 1]))
                hb[h][1] += 1
                hc[h][1] += 1
                hb[h][0] += eb < R
                hc[h][0] += ec < R
    for h in HOR:
        print('binned-oracle@%d: %.1f%%  continuous-oracle@%d: %.1f%%' % (
            h, 100.0 * hb[h][0] / hb[h][1], h, 100.0 * hc[h][0] / hc[h][1]), flush=True)


if __name__ == '__main__':
    main()
