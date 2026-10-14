"""recover_viterbi.py — windowed Viterbi decode of held keys.

Per-tick inversion fails on ~67% of ticks: at max speed acceleration is ~0, so
a single tick carries almost no information about which key is held (forward and
backward taps are indistinguishable). Keys are HELD across consecutive ticks, so
decode the whole window jointly.

For each tick we simulate all 9 yaw-relative candidates (8 compass directions
rotated by the player's facing + coast) through the real engine physics with
map collision. cost[t][c] = ||v_sim - v_obs||. A Viterbi pass over the window
finds the globally best explanation: high-information ticks (from rest, or
direction changes) anchor the decode and low-information ticks inherit from
their neighbours.

FLIP_PENALTY discourages one-tick direction flips (measured 0.34/tick with a
per-tick decoder; human play is ~0.02-0.1/tick).

Output: discrete label per tick + honest diagnostics (residual, ambiguity,
flip rate).
"""
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (AIRCAP, BACK_CLAMP, BACK_CLAMP_MIN, DT, FL_DUCKING,
                     FL_ONGROUND, GRAVITY, accelerate, air_accelerate,
                     friction, load_csv_live, yaw_to_fwd)
from boxtrace import HULL_DUCK, HULL_STAND, grounded_box, load_brushes, move_box

BASE = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
NSTATE = 9      # 8 directions + coast
COAST = 8
INF = 1e18


KEY_SPEED = 450.0   # cl_forwardspeed/cl_sidespeed: ALWAYS 450 per axis, class-independent
# TF2 class base speeds (tf_classdata). The demo's mspd column is UNRELIABLE
# (observed 0/80/110/160/230/240, including 240 for a 110-speed class), so the
# class from the demo is the authority.
CLASS_SPEED = {1: 400.0, 2: 300.0, 3: 240.0, 4: 280.0, 5: 320.0,
               6: 300.0, 7: 300.0, 8: 300.0, 9: 300.0}


def step(p, v, ci, fwd, maxspeed, on_ground, ducking, BR):
    """Velocity after one tick holding candidate ci.

    Follows CTFGameMovement::WalkMove / AirMove (tf_gamemovement.cpp:1747/2070)
    EXACTLY:
      wishvel = forward*fm + right*sm        (fm/sm are +450/-450 per key)
      wishdir = normalize(wishvel)
      wishspeed = |wishvel|, clamped to maxspeed   <-- DIAGONAL IS 450*sqrt2
      Accelerate(wishdir, wishspeed, sv_accelerate)
      then clamp |velocity_xy| to maxspeed, then the BACK-SPEED clamp.

    The earlier bug collapsed the diagonal to min(450,maxspeed) instead of
    min(450*sqrt2, maxspeed), and skipped the back clamp; both change the
    predicted velocity enough to break the match.
    """
    half = HULL_DUCK if ducking else HULL_STAND
    # per-axis forward/side amounts from the candidate's (bx, by)
    if ci == COAST:
        fm = sm = 0.0
    else:
        bx, by = BASE[ci]
        fm = bx * KEY_SPEED
        sm = by * KEY_SPEED
    right = np.array([fwd[1], -fwd[0]])
    wishvel = fwd * fm + right * sm                      # 2D
    mag = float(np.linalg.norm(wishvel))
    if mag > 1e-9:
        wishdir = wishvel / mag
        wishspeed = min(mag, maxspeed)                   # clamp AFTER normalize
    else:
        wishdir = None
        wishspeed = 0.0
    # accelerate() dots against the full 3-vector velocity; keep wishdir 3D.
    wdir3 = None if wishdir is None else np.array([wishdir[0], wishdir[1], 0.0])

    if on_ground:
        v2 = np.array([v[0], v[1], 0.0])
        v2 = friction(v2)
        if wishdir is not None:
            v2[:2] = accelerate(v2[:2], wdir3[:2], wishspeed)
        # CheckVelocity: clamp XY speed to maxspeed
        sp = float(np.linalg.norm(v2[:2]))
        if sp > maxspeed and sp > 1e-9:
            v2[:2] *= maxspeed / sp
        # back-speed clamp (tf_clamp_back_speed=0.9, min=100)
        if BACK_CLAMP < 1.0 and sp > BACK_CLAMP_MIN:
            flDot = float(np.dot(fwd, v2[:2]))
            if flDot < 0:
                backm = fwd * flDot
                rightm = right * float(np.dot(right, v2[:2]))
                bs = float(np.linalg.norm(backm))
                mb = maxspeed * BACK_CLAMP
                if bs > mb and bs > 1e-9:
                    backm = backm * (mb / bs)
                v2[:2] = backm + rightm
                sp2 = float(np.linalg.norm(v2[:2]))
                if sp2 > maxspeed and sp2 > 1e-9:
                    v2[:2] *= maxspeed / sp2
    else:
        v2 = v.copy()
        if wishdir is not None:
            v2[:2] = air_accelerate(v2[:2], wdir3[:2], wishspeed, AIRCAP)
        v2[2] -= GRAVITY * DT
    pend, _h, _n, _u = move_box(p, p + v2 * DT, half, BR)
    return (pend - p) / DT, pend


def decode_window(P, V, FLAGS, EY, CAPS, BR, start, length, flip_pen=1.0):
    """Viterbi over [start, start+length). Returns (labels, resid, ambig)."""
    labels = np.full(length, -1, dtype=np.int8)
    resid = np.full(length, np.nan, dtype=np.float32)
    ambig = np.full(length, False, dtype=bool)
    dp = np.full(NSTATE, INF, dtype=np.float64)
    bp = np.full((length, NSTATE), -1, dtype=np.int8)
    cost = np.full((length, NSTATE), INF, dtype=np.float64)
    valid = np.zeros(length, dtype=bool)
    pen = np.zeros((NSTATE, NSTATE))
    if flip_pen:
        for a in range(NSTATE):
            for b in range(NSTATE):
                pen[a, b] = 0.0 if a == b else flip_pen
    for t in range(length):
        k = start + t
        if k + 1 >= len(P):
            break
        p = P[k]
        v = V[k]
        vn = V[k + 1]
        fl = FLAGS[k]
        on_ground = bool(fl & FL_ONGROUND)
        ducking = bool(fl & FL_DUCKING)
        # maxspeed from the CLASS, not the demo's mspd column: observed p95
        # speed per class == CLASS_SPEED exactly (1->400, 2->300, 3->240,
        # 5->320, 7->300). mspd is uniq 0/80/110/160/230/240 and is already
        # duck-adjusted, so re-applying DUCK_FRAC quartered it and made
        # friction brake the player 240->70 instead of holding ~191.
        ms = CAPS[k]
        fwd = yaw_to_fwd(EY[k])
        for c in range(NSTATE):
            pv, _pend = step(p, v, c, fwd, ms, on_ground, ducking, BR)
            cost[t, c] = float(np.linalg.norm(pv - vn))
        valid[t] = True
        if t == 0:
            bp[t, :] = COAST        # free start state
            dp[:] = cost[t, :]
            continue
        tot = dp[:, None] + pen + cost[t, :][None, :]
        best_prev = np.argmin(tot, axis=0)
        bp[t] = best_prev.astype(np.int8)
        dp = tot[best_prev, np.arange(NSTATE)]
    n = int(valid.sum())
    if n == 0:
        return labels, resid, ambig
    state = int(np.argmin(dp))
    path = np.zeros(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = int(bp[t, state]) if bp[t, state] >= 0 else COAST
    labels[:n] = path
    resid[:n] = cost[np.arange(n), path]
    # ambiguity: best vs second-best cost at this tick
    srt = np.sort(cost[:n], axis=1)
    ambig[:n] = (srt[:, 1] - srt[:, 0]) < 0.5
    return labels, resid, ambig


def main():
    csvpath = sys.argv[1]
    brushpath = sys.argv[2]
    nwin = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    wlen = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    flip = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    BR = load_brushes(brushpath)
    print('brushes=%d' % len(BR['bnsides']), flush=True)
    rows = load_csv_live(csvpath, classes=set(range(1, 10)))
    e = max(rows, key=lambda x: len(rows[x]))
    rs = rows[e]
    P = np.array([[float(r['ox']), float(r['oy']), float(r['oz'])] for r in rs])
    V = np.array([[float(r['vx']), float(r['vy']), float(r['vz'])] for r in rs])
    FLAGS = [int(float(r['flags'])) for r in rs]
    EY = [float(r['eyey']) for r in rs]
    MSPD = [float(r.get('mspd') or 300.0) for r in rs]
    # Effective maxspeed from the CLASS, not mspd. Verified against this demo:
    # observed p95 speed per class == CLASS_SPEED exactly (1->400, 2->300,
    # 3->240, 5->320, 7->300). The mspd column is unreliable (uniq
    # 0/80/110/160/230/240, incl. 240 for the 110-speed Engineer).
    CLS = [int(float(r['class'])) for r in rs]
    CAPS = [CLASS_SPEED.get(c, 300.0) for c in CLS]
    TICKS = [int(r['tick']) for r in rs]
    allr, alla, allf = [], [], []
    for w in range(nwin):
        s = 200 + w * wlen
        if s + wlen + 1 >= len(rs):
            break
        if any(TICKS[i + 1] - TICKS[i] != 1 for i in range(s, s + wlen)):
            continue
        lab, res, amb = decode_window(P, V, FLAGS, EY, CAPS, BR, s, wlen, flip)
        m = ~np.isnan(res)
        if not m.any():
            continue
        allr.append(res[m])
        alla.append(amb[m])
        allf.append(float(np.mean(lab[m][1:] != lab[m][:-1])))
        if w < 3:
            print('  win%d start=%d resid med=%.4f ambig=%.2f%% flips=%.3f  labels=%s'
                  % (w, s, np.median(res[m]), 100 * amb[m].mean(),
                     allf[-1], lab[m][:16].tolist()), flush=True)
    if allr:
        R = np.concatenate(allr)
        A = np.concatenate(alla)
        print('\nWINDOWS=%d ticks=%d  W=%d flip_pen=%.1f' % (len(allr), len(R), wlen, flip))
        print('  residual  med %.4f  p90 %.4f  p99 %.4f  (STV floor ~0.67 hu/s)' %
              (np.median(R), np.percentile(R, 90), np.percentile(R, 99)))
        print('  frac ticks residual < 1hu/s: %.4f' % (R < 1).mean())
        print('  ambiguous ticks: %.3f%%' % (100 * A.mean()))
        print('  mean flips/tick: %.4f  (human ~0.02-0.10)' % np.mean(allf))


if __name__ == '__main__':
    main()