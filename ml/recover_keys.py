"""recover_keys.py — recover the exact key press from motion, by simulation.

SourceTV demos do not store CLC_Move (verified: 200k frames, zero dem_usercmd;
the only WriteUserCmd caller is the CLIENT recorder, cl_main.cpp:2175). So the
keys must be inferred. But the inference is NOT a continuous optimisation:
TF2 movement keys are digital WASD, so each tick the player held exactly one of
9 yaw-relative directions (8 + coast). We simulate all 9 through the real
engine physics (with map collision) and keep the one that reproduces the
observed velocity.

Per tick we know: position, velocity, grounded, ducking, and the player's
facing (yaw) exactly — STV replicates m_angEyeAngles (cs_player.cpp:367,
SendPropAngle SPROP_CHANGES_OFTEN), which is why per-tick yaw is exact here.

Candidates are yaw-relative: the 8 compass directions rotated by the player's
facing, plus coast. Wish direction is in WORLD frame after rotation, so this
matches ComputeWishVelocity (forward = yaw dir, right = yaw dir - 90 deg).

Output: discrete label per tick (0 = coast, 1..8 = N,NE,E,SE,S,SW,W,NW in yaw
frame), plus residual stats and an ambiguity count. Ambiguous ticks (two
candidates within TIE of each other) are flagged, not hidden.
"""
import math
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (ACCEL, AIRACCEL, AIRCAP, DUCK_FRAC, DT, FL_DUCKING,
                     FL_ONGROUND, GRAVITY, SF, STOPSPEED, WISH_THRESH,
                     air_accelerate, accelerate, back_clamp, friction,
                     ground_step, load_csv_live, yaw_to_fwd)
from boxtrace import HULL_DUCK, HULL_STAND, grounded_box, load_brushes, move_box

# 8 compass dirs in yaw frame: (forward, right) unit combos, ordered
# N, NE, E, SE, S, SW, W, NW relative to the player's facing.
BASE = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
TIE = 0.05      # hu/s: candidates this close are ambiguous


def tick_step(p, v, wishdir, wishspeed, maxspeed, on_ground, ducking, BR, fwd):
    """One engine tick, with collision.

    Uses the SAME primitives the verified inverse uses, so replaying the
    inverted input reproduces the demo exactly (residual 0.0000). A hand-rolled
    friction+accelerate+maxspeed-clamp version was wrong: the clamp alone drove
    p90 residual to 745 hu/s. ground_step is called with maxspeed=1e9 for the
    same "no 2D clamp" reason inverse_ground uses it."""
    half = HULL_DUCK if ducking else HULL_STAND
    if on_ground:
        v2 = np.array([v[0], v[1], 0.0], dtype=float)
        if wishdir is not None:
            v2, _stopped = ground_step(v2, wishdir, wishspeed, 1e9)
        else:
            v2, _stopped = ground_step(v2, np.array([1.0, 0.0]), 0.0, 1e9)
    else:
        v2 = v.copy()
        v2[2] -= GRAVITY * DT
        if wishdir is not None:
            v2[:2] = air_accelerate(v2[:2], wishdir, wishspeed, AIRCAP, AIRACCEL)
    pend, _h, _n, _u = move_box(p, p + v2 * DT, half, BR)
    return pend, (pend - p) / DT


def recover_entity(rs, BR, maxspeed, tstart=0, tend=None):
    """Recover the key label for each tick of one entity's consecutive run."""
    labels = np.zeros(len(rs), dtype=np.int8)
    resid = np.full(len(rs), np.nan, dtype=np.float32)
    ambig = np.zeros(len(rs), dtype=bool)
    errs = []
    n = len(rs)
    if tend is None:
        tend = n
    for k in range(tstart, min(tend, n) - 1):
        try:
            p = np.array([float(rs[k]['ox']), float(rs[k]['oy']), float(rs[k]['oz'])])
            v = np.array([float(rs[k]['vx']), float(rs[k]['vy']), float(rs[k]['vz'])])
            pn = np.array([float(rs[k + 1]['ox']), float(rs[k + 1]['oy']),
                           float(rs[k + 1]['oz'])])
            vn = np.array([float(rs[k + 1]['vx']), float(rs[k + 1]['vy']),
                           float(rs[k + 1]['vz'])])
            ey = float(rs[k]['eyey'])
            fl = int(float(rs[k]['flags']))
            mspd = float(rs[k].get('mspd') or 0.0)
        except (ValueError, KeyError, IndexError):
            continue
        # consecutive tick check (poison rule)
        try:
            if int(rs[k + 1]['tick']) - int(rs[k]['tick']) != 1:
                continue
        except (ValueError, KeyError):
            continue
        duck = bool(fl & FL_DUCKING)
        on_ground = bool(fl & FL_ONGROUND)
        # class maxspeed from the demo itself; hardcoding 300 made every
        # acceleration wrong for any non-scout class (residual 22hu/s).
        ms = mspd if mspd > 1.0 else 300.0
        if duck:
            ms *= DUCK_FRAC
        fwd = yaw_to_fwd(ey)
        # candidate wish directions, rotated by facing
        cands = []
        for bx, by in BASE:
            wd = bx * fwd + by * np.array([fwd[1], -fwd[0]])
            n = float(np.linalg.norm(wd))
            cands.append(wd / n if n > 1e-9 else None)
        cands.append(None)  # coast
        best = None
        best_e = 1e18
        second = 1e18
        for ci, wd in enumerate(cands):
            _pp, pv = tick_step(p, v, wd, ms if wd is not None else 0.0,
                                ms, on_ground, duck, BR, fwd)
            e = float(np.linalg.norm(pv - vn))
            errs.append(e)
            if e < best_e:
                second = best_e
                best_e = e
                best = ci
            elif e < second:
                second = e
        labels[k] = best
        resid[k] = best_e
        ambig[k] = (second - best_e) < TIE
    return labels, resid, ambig


def main():
    csvpath = sys.argv[1]
    brushpath = sys.argv[2]
    nent = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    ntick = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    BR = load_brushes(brushpath)
    print('brushes=%d' % len(BR['bnsides']), flush=True)
    rows = load_csv_live(csvpath, classes=set(range(1, 10)))
    ents = sorted(rows, key=lambda e: -len(rows[e]))[:nent]
    allres, allamb, allflips = [], [], []
    for e in ents:
        rs = rows[e]
        labels, resid, ambig = recover_entity(rs, BR, 300.0, 0, min(ntick, len(rs)))
        m = ~np.isnan(resid)
        if m.sum() == 0:
            continue
        r = resid[m]
        am = ambig[m]
        lab = labels[m]
        flips = float(np.mean(lab[1:] != lab[:-1]))
        allres.append(r)
        allamb.append(am)
        allflips.append(flips)
        print('ent%-3d ticks=%-6d resid med=%.4f p90=%.4f p99=%.4f  ambig=%.3f%%  flips=%.4f'
              % (e, int(m.sum()), np.median(r), np.percentile(r, 90),
                 np.percentile(r, 99), 100.0 * am.mean(), flips), flush=True)
    if allres:
        R = np.concatenate(allres)
        A = np.concatenate(allamb)
        print('\nOVERALL ticks=%d' % len(R))
        print('  median residual %.4f hu/s  (STV quantization floor ~0.67)' % np.median(R))
        print('  p90 %.4f  p99 %.4f' % (np.percentile(R, 90), np.percentile(R, 99)))
        print('  ambiguous ticks: %.3f%%' % (100.0 * A.mean()))
        print('  mean direction flips per tick: %.4f (human ~= 0.02-0.1)' % np.mean(allflips))


if __name__ == '__main__':
    main()