"""eval_cont_wall.py — honest champion eval: real collision sim, no future leak.

Replaces eval_cont.py's free-space rollout + future MSK ground-flag cheat with
the same engine-faithful path the oracle uses (ml/oracle_tick.py tick_player):
friction -> accelerate / air-accelerate -> swept-box move with slide + step-up
-> velocity from actual displacement -> ground trace.

Ground state is DERIVED from the sim (grounded_box), never read from the demo's
future flags. Jump is never predicted: if the model's window contains a real
jump, the window is scored with the demo's own ground/air transition applied as
a given event (same convention as the oracle), so jump performance is not
attributed to input prediction.

Usage:
  python ml/eval_cont_wall.py <ckpt.pt> <brush_solid.npz> <eval1.npz> [...]
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from boxtrace import load_brushes
from movesim import (AIRCAP, BACK_CLAMP, BACK_CLAMP_MIN, DT, FL_DUCKING,
                     FL_ONGROUND, GRAVITY, accelerate, air_accelerate,
                     back_clamp, friction, yaw_to_fwd)
import train_cont
from train_cont import HIST, FUT

HOR = [7, 13, 20, 66, 132]
R = 24.0


def tick(p, v, wishdir, wishspeed, maxspeed, on_ground, ducking, half, BR, fwd):
    from boxtrace import grounded_box
    v = np.asarray(v, dtype=float).copy()
    if on_ground:
        v[:2] = friction(v[:2])
        v[2] = 0.0
        if wishdir is not None:
            v[:2] = accelerate(v[:2], wishdir, wishspeed)
        sp = float(np.linalg.norm(v[:2]))
        if sp > maxspeed and sp > 1e-9:
            v[:2] *= maxspeed / sp
        if wishdir is not None and wishspeed > 40.0:
            v[:2] = back_clamp(v[:2], fwd, maxspeed)
            sp = float(np.linalg.norm(v[:2]))
            if sp > maxspeed and sp > 1e-9:
                v[:2] *= maxspeed / sp
    else:
        v[2] -= GRAVITY * DT
        if wishdir is not None:
            v[:2] = air_accelerate(v[:2], wishdir, wishspeed, AIRCAP)
    pend, _h, _n, _u = _mb(p, p + v * DT, half, BR)
    v = (pend - p) / DT
    return pend, v, grounded_box(pend, half, BR)


def _mb(p0, p1, half, BR):
    from boxtrace import move_box
    return move_box(p0, p1, np.asarray(half, dtype=float), BR)


def main():
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    if 'cfg' in ck:
        train_cont.D, train_cont.NL, train_cont.NH, train_cont.FF = ck['cfg'][:4]
        if len(ck['cfg']) > 6:
            train_cont.NFEAT = ck['cfg'][6]
    net = train_cont.ContAR()
    net.load_state_dict(ck['net'])
    net.eval()
    torch.set_num_threads(4)
    BR = load_brushes(sys.argv[2])
    files = sys.argv[3:]
    from boxtrace import HULL_DUCK, HULL_STAND
    stats = {k: {h: [0, 0] for h in HOR} for k in ('coast', 'model')}
    for path in files:
        d = np.load(path, allow_pickle=True)
        X, POS = d['X'], d['POS']
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
                    w, sp, _j, _c = net(xb, fin, jin.long())
                    wk = w[:, k].numpy()
                    sk = sp[:, k].numpy()
                    nl = np.linalg.norm(wk, axis=1, keepdims=True)
                    pw[:, k] = np.where(nl > 1e-6, wk / np.maximum(nl, 1e-6), 0.0)
                    ps[:, k] = sk
                    if k + 1 < FUT:
                        fin[:, k + 1, :2] = torch.tensor(pw[:, k])
                        fin[:, k + 1, 2] = torch.tensor(ps[:, k])
            for i in range(n):
                gi = s + i
                ey0 = math.degrees(math.atan2(X[gi, HIST - 1, 7], X[gi, HIST - 1, 8]))
                p0 = X[gi, HIST - 1, 0:3] * 1000.0
                v0 = X[gi, HIST - 1, 3:6] * 1000.0
                ms = X[gi, HIST - 1, 9] * 400.0
                fwd = yaw_to_fwd(ey0)
                truth = POS[gi] + p0
                for kind in ('coast', 'model'):
                    p = p0.copy()
                    v = v0.copy()
                    gr = bool(X[gi, HIST - 1, 6] > 0.5)
                    traj = {}
                    for k in range(FUT):
                        duck = False
                        half = HULL_DUCK if duck else HULL_STAND
                        if kind == 'model':
                            wd = pw[i, k]
                            spd = float(ps[i, k]) * 400.0
                            if np.linalg.norm(wd) <= 1e-9 or spd <= 25.0:
                                wd, spd = None, 0.0
                        else:
                            wd, spd = None, 0.0
                        p, v, gr2 = tick(p, v, wd, spd, ms, gr, duck, half, BR, fwd)
                        gr = gr2
                        traj[k] = p.copy()
                    for h in HOR:
                        if h > FUT:
                            continue
                        e = float(np.linalg.norm(traj[h - 1] - truth[h - 1]))
                        stats[kind][h][0] += e < R
                        stats[kind][h][1] += 1
    print('%-8s %s' % ('ctrl', '  '.join('h=%-4d' % h for h in HOR)))
    for kind in ('coast', 'model'):
        row = []
        for h in HOR:
            hit, tot = stats[kind][h]
            row.append('%5.1f%%' % (100.0 * hit / max(tot, 1)) if tot else '  -  ')
        print('%-8s %s' % (kind, '  '.join(row)))
    print('\nNOTE: collision sim, ground derived from sim (no future flags), '
          'R=%.0f, jump not predicted.' % R)


if __name__ == '__main__':
    main()