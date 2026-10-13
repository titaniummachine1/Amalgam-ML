"""oracle_tick.py — tick-exact engine replay WITH collision (the real sim).

One question: if we know the player's inputs for the next 1-2 seconds, does our
simulated position match the demo? This is the path that will run in Amalgam:

  friction -> accelerate / air_accelerate+gravity -> swept-box move (bump,
  clip-velocity, step-up) -> velocity from actual displacement -> ground trace

Tick order follows game/shared/tf/tf_gamemovement.cpp PlayerMove +
CheckVelocity. Jump is a GIVEN event from the demo, never predicted.
Scope = horizontal input + camera yaw/pitch.

Usage: python ml/oracle_tick.py <csv> <brush.npz> [nwin] [fut]
"""
import math
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (AIRACCEL, AIRCAP, ACCEL, DUCK_FRAC, DT, FL_DUCKING,
                     FL_ONGROUND, GRAVITY, WISH_THRESH, air_accelerate,
                     accelerate, back_clamp, friction, yaw_to_fwd)
from boxtrace import HULL_DUCK, HULL_STAND, grounded_box, load_brushes, move_box
from gate_replay import build_windows, recover_inputs

SPLIT = 66


def tick_player(p, v, wishdir, wishspeed, maxspeed, on_ground, ducking, BR, fwd):
    """One engine tick of horizontal movement with collision.
    Returns (pos, vel, grounded_after).

    Order (tf_gamemovement.cpp PlayerMove): friction -> accelerate / air move
    (+gravity) -> CheckVelocity (clamp) -> category move with collision ->
    FinishMove (velocity from actual displacement, ground trace)."""
    half = HULL_DUCK if ducking else HULL_STAND
    v = np.asarray(v, dtype=float).copy()
    if on_ground:
        v[:2] = friction(v[:2])
        v[2] = 0.0
        if wishdir is not None:
            v[:2] = accelerate(v[:2], wishdir, wishspeed, ACCEL)
    else:
        v[2] -= GRAVITY * DT
        if wishdir is not None:
            v[:2] = air_accelerate(v[:2], wishdir, wishspeed, AIRCAP, AIRACCEL)
    hs = float(np.linalg.norm(v[:2]))
    if hs > maxspeed:
        v[:2] *= maxspeed / hs
    if on_ground and wishdir is not None and wishspeed > WISH_THRESH:
        v[:2] = back_clamp(v[:2], fwd, maxspeed)
    pend, _hit, _nrm, _nud = move_box(p, p + v * DT, half, BR)
    v = (pend - p) / DT
    return pend, v, bool(grounded_box(pend, half, BR))


def main():
    csvpath = sys.argv[1]
    brushpath = sys.argv[2]
    nwin = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    fut = int(sys.argv[4]) if len(sys.argv) > 4 else 132
    BR = load_brushes(brushpath)
    if '--nocollision' in sys.argv:
        BR = dict(bplanes=np.zeros((0, 6, 4)), bnsides=np.array([], dtype=np.int64),
                  bcontents=np.array([], dtype=np.uint32),
                  bmins=np.zeros((0, 3)), bmaxs=np.zeros((0, 3)))
        print('COLLISION DISABLED (bisect mode)', flush=True)
    print('brushes=%d' % len(BR['bnsides']), flush=True)
    ego, wins = build_windows(csvpath, None, None, nwin, horizon=fut)
    HOR = [h for h in (7, 13, 20, 33, 66, 100, 132) if h <= fut]
    errs = {h: [] for h in HOR}
    prof = np.zeros(fut)
    nwin_used = 0
    for tick, cls, ms0, P, V, flags, eyey, ents in wins:
        rec = recover_inputs(V, flags, ms0, SPLIT, SPLIT + fut)
        p = P[SPLIT].copy().astype(float)
        v = V[SPLIT].copy().astype(float)
        fwd = yaw_to_fwd(eyey[SPLIT])
        traj = np.zeros((fut, 3))
        for k in range(fut):
            wd, ws = rec[k]
            duck = bool(flags[SPLIT + k] & FL_DUCKING)
            ms = ms0 * (DUCK_FRAC if duck else 1.0)
            gr = bool(flags[SPLIT + k] & FL_ONGROUND)
            p, v, _g = tick_player(p, v, wd, ws, ms, gr, duck, BR, fwd)
            traj[k] = p
        e = np.linalg.norm(traj - P[SPLIT + 1:SPLIT + fut + 1], axis=1)
        prof += e
        for h in HOR:
            errs[h].append(e[h - 1])
        nwin_used += 1
    prof /= max(nwin_used, 1)
    print('windows=%d fut=%d ego=ent%d' % (nwin_used, fut, ego))
    print('\n%-6s %-12s %-10s %-10s %s' % ('tick', 'hit<24hu', 'med-err', 'p90-err', 'max-err'))
    for h in HOR:
        a = np.array(errs[h])
        print('%-6d %-12.1f %-10.3f %-10.3f %.3f' %
              (h, 100.0 * (a < 24).mean(), np.median(a), np.percentile(a, 90), a.max()))
    print('\nper-tick median error:')
    for h in HOR:
        print('  tick %3d (%.2fs): %10.4f hu' % (h, h * DT, prof[h - 1]))
    print('\nEXACT (<1hu) / (<5hu) fraction:')
    for h in HOR:
        a = np.array(errs[h])
        print('  tick %3d: <1hu %5.1f%%   <5hu %5.1f%%' %
              (h, 100.0 * (a < 1).mean(), 100.0 * (a < 5).mean()))


if __name__ == '__main__':
    main()