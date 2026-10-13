"""probe_signal.py — is there ANY predictable signal over coast?
Compares, on identical windows from the raw CSV:
  coast      : no input at all
  hold-last  : repeat the last observed input forever (intent persistence)
  oracle     : per-tick recovered inputs (ceiling)
If hold-last <= coast, no intent model can win and training is pointless.
Usage: python ml/probe_signal.py <csv> [nwin]
"""
import math
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import AIRCAP, rollout_free, yaw_to_fwd
from gate_replay import build_windows, recover_inputs

HOR = [7, 13, 20, 66]
R = 24.0


def main():
    csvpath = sys.argv[1]
    nwin = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    ego, wins = build_windows(csvpath, None, None, nwin, horizon=66)
    st = {}
    res = {k: {h: [0, 0, []] for h in HOR} for k in ('coast', 'hold', 'oracle')}
    for tick, cls, ms0, P, V, flags, eyey, ents in wins:
        split, fut = 66, 66
        p0, v0, ey0 = P[split], V[split], eyey[split]
        views = [yaw_to_fwd(ey0)] * fut
        grf = [bool(flags[split + k] & 1) for k in range(fut)]
        rec = recover_inputs(V, flags, ms0, split, split + fut)
        last = None
        for wd, ws in rec:
            if wd is not None:
                last = (wd, ws)
        hold = [last if last is not None else (None, 0.0)] * fut
        sets = {'coast': [(None, 0.0)] * fut, 'hold': hold, 'oracle': rec}
        for k, inp in sets.items():
            pr = rollout_free(v0, p0, fut, inp, grf, ms0, AIRCAP, views)
            for h in HOR:
                e = float(np.linalg.norm(pr[h - 1] - P[split + h]))
                res[k][h][0] += e < R
                res[k][h][1] += 1
                res[k][h][2].append(e)
    print('windows=%d ego=ent%d  R=%.0f' % (len(wins), ego, R))
    print('%-8s %s' % ('ctrl', '  '.join('h=%-14d' % h for h in HOR)))
    for k in ('coast', 'hold', 'oracle'):
        row = []
        for h in HOR:
            hit, tot, e = res[k][h]
            row.append('%5.1f%%/%.1f ' % (100.0 * hit / max(tot, 1), np.median(e)))
        print('%-8s %s' % (k, '  '.join(row)))
    print('\nVERDICT: %s' % (
        'signal exists (hold-last > coast) -> intent is persistent, model can win'
        if res['hold'][20][0] / max(res['hold'][20][1], 1) >
           res['coast'][20][0] / max(res['coast'][20][1], 1)
        else 'NO SIGNAL over coast at h=20 -> a model cannot win here'))


if __name__ == '__main__':
    main()