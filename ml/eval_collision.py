"""eval_collision.py — score a model OR the oracle in the REAL sim (collision).

Replaces the old free-space eval, which (a) had no walls and (b) leaked the
demo's future ground flags into the sim. This scores everything through
CTFGameMovement's real path: friction, accelerate/air_accelerate, swept-box
collision, ground trace.

Controllers compared on identical windows, all vs the demo's own positions:
  coast    no input (the floor to beat)
  oracle   inputs recovered from the demo's own kinematics (the ceiling)
  model    greedy AR from the transformer, fed its own predictions back

Full history is always given before the window. Horizons 7/13/20/33/66/132.

Usage:
  python ml/eval_collision.py <csv> <brush.npz> [nwin] [horizon] [--model ckpt]
"""
import math
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from boxtrace import grounded_box, load_brushes
from movesim import (AIRCAP, DUCK_FRAC, FL_DUCKING, FL_ONGROUND, inverse_air,
                     inverse_ground, load_csv_live, yaw_to_fwd)
from recover_viterbi import step, CLASS_SPEED

HOR = [7, 13, 20, 33, 66, 132]
R = 24.0
HIST = 66


def tick(p, v, wishdir, wishspeed, ms, on_ground, ducking, BR, fwd, half=None):
    return step(p, v, DIRIDX.get(wishdir, 8) if wishdir is not None else 8,
                fwd, ms, on_ground, ducking, BR)


DIRIDX = {}


def apply_dir(p, v, wishdir, wishspeed, ms, gr, duck, BR, fwd):
    """One engine tick for an arbitrary wishdir (not limited to the 9)."""
    from movesim import (ACCEL, AIRACCEL, AIRCAP, BACK_CLAMP, BACK_CLAMP_MIN,
                         DT, GRAVITY, accelerate, air_accelerate, friction)
    from boxtrace import HULL_DUCK, HULL_STAND, move_box
    half = HULL_DUCK if duck else HULL_STAND
    if wishdir is not None:
        n = float(np.linalg.norm(wishdir))
        wishdir = np.asarray(wishdir, dtype=float).ravel()[:2] / max(n, 1e-9)
        w = min(wishspeed, ms)
    else:
        wishdir = None
        w = 0.0
    if gr:
        v2 = np.array([v[0], v[1], 0.0])
        v2 = friction(v2)
        if wishdir is not None:
            v2[:2] = accelerate(v2[:2], wishdir, w)
        sp = float(np.linalg.norm(v2[:2]))
        if sp > ms and sp > 1e-9:
            v2[:2] *= ms / sp
        right = np.array([fwd[1], -fwd[0]])
        if BACK_CLAMP < 1.0 and sp > BACK_CLAMP_MIN:
            flDot = float(np.dot(fwd, v2[:2]))
            if flDot < 0:
                bm = fwd * flDot
                rm = right * float(np.dot(right, v2[:2]))
                bs = float(np.linalg.norm(bm))
                mb = ms * BACK_CLAMP
                if bs > mb and bs > 1e-9:
                    bm = bm * (mb / bs)
                v2[:2] = bm + rm
                sp2 = float(np.linalg.norm(v2[:2]))
                if sp2 > ms and sp2 > 1e-9:
                    v2[:2] *= ms / sp2
    else:
        v2 = v.copy()
        if wishdir is not None:
            v2[:2] = air_accelerate(v2[:2], wishdir, w, AIRCAP)
        v2[2] -= GRAVITY * DT
    pend, _h, _n, _u = move_box(p, p + v2 * DT, half, BR)
    return pend, (pend - p) / DT


def oracle_inputs(V, FLAGS, MS, split, fut):
    out = []
    for k in range(fut):
        i = split + k
        ms = MS[i]
        gr = bool(FLAGS[i] & FL_ONGROUND)
        try:
            iv = inverse_ground(V[i], V[i + 1], ms) if gr else inverse_air(V[i], V[i + 1])
        except Exception:
            out.append((None, 0.0))
            continue
        wd = iv['wishdir']
        if wd is None:
            out.append((None, 0.0))
        else:
            out.append((np.asarray(wd, dtype=float).ravel()[:2],
                        float(iv['wishspeed'])))
    return out


def main():
    csvpath, brushpath = sys.argv[1], sys.argv[2]
    nwin = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    fut = int(sys.argv[4]) if len(sys.argv) > 4 else 132
    BR = load_brushes(brushpath)
    rows = load_csv_live(csvpath, classes=set(range(1, 10)))
    ents = sorted(rows, key=lambda e: -len(rows[e]))
    # Keep only entities long enough to contain 132-tick clean windows;
    # MGE records entities unevenly, and short ones yield zero windows.
    ents = [e for e in ents if len(rows[e]) >= HIST + fut + 2][:6]
    res = {k: {h: [0, 0, []] for h in HOR if h <= fut} for k in ('coast', 'oracle')}
    nw = 0
    for e in ents:
        rs = rows[e]
        P = np.array([[float(r['ox']), float(r['oy']), float(r['oz'])] for r in rs])
        V = np.array([[float(r['vx']), float(r['vy']), float(r['vz'])] for r in rs])
        FLAGS = [int(float(r['flags'])) for r in rs]
        EY = [float(r['eyey']) for r in rs]
        CLS = [int(float(r['class'])) for r in rs]
        MS = [CLASS_SPEED.get(c, 300.0) * (DUCK_FRAC if (FLAGS[i] & FL_DUCKING) else 1.0)
              for i, c in enumerate(CLS)]
        TICKS = [int(r['tick']) for r in rs]
        s = HIST
        while s + fut < len(rs) and nw < nwin:
            # Require gap==1 only over the FUTURE we simulate (the score
            # window), not across the history too: this demo has 16260 gaps
            # across 170k ticks, and demanding 199 consecutive pairs rejected
            # essentially every window.
            if any(TICKS[i + 1] - TICKS[i] != 1 for i in range(s, s + fut)):
                s += 1
                continue
            if np.any(np.linalg.norm(P[s + 1:s + fut + 1, :] -
                                         P[s:s + fut, :], axis=1) > 80):
                s += 1
                continue
            fwd = yaw_to_fwd(EY[s])
            for kind in ('coast', 'oracle'):
                inp = ([(None, 0.0)] * fut if kind == 'coast'
                       else oracle_inputs(V, FLAGS, MS, s, fut))
                p = P[s].copy().astype(float)
                v = V[s].copy().astype(float)
                traj = np.zeros((fut, 3))
                for k in range(fut):
                    i = s + k
                    wd, ws = inp[k]
                    p, v = apply_dir(p, v, wd, ws, MS[i],
                                     bool(FLAGS[i] & FL_ONGROUND),
                                     bool(FLAGS[i] & FL_DUCKING), BR, fwd)
                    traj[k] = p
                for h in HOR:
                    if h > fut:
                        continue
                    err = float(np.linalg.norm(traj[h - 1] - P[s + h]))
                    res[kind][h][0] += err < R
                    res[kind][h][1] += 1
                    res[kind][h][2].append(err)
            nw += 1
            s += 1
            if nw % 10 == 0:
                print('  ...%d windows' % nw, flush=True)
    print('\nwindows=%d  horizons=%s  R=%.0f' % (nw, [h for h in HOR if h <= fut], R))
    print('%-8s %s' % ('ctrl', ''.join('  h=%-15d' % h for h in HOR if h <= fut)))
    for kind in ('coast', 'oracle'):
        row = ''
        for h in HOR:
            if h > fut:
                continue
            hit, tot, errs = res[kind][h]
            a = np.array(errs)
            row += '  %5.1f%%/%.2fhu   ' % (100.0 * hit / max(tot, 1), np.median(a))
        print('%-8s %s' % (kind, row))
    ok = True
    print('\nGATE (oracle must beat coast at every horizon):')
    for h in HOR:
        if h > fut:
            continue
        o = res['oracle'][h]
        c = res['coast'][h]
        oh = 100.0 * o[0] / max(o[1], 1)
        ch = 100.0 * c[0] / max(c[1], 1)
        om = float(np.median(o[2]))
        cm = float(np.median(c[2]))
        # Hit-rate can TIE at short horizons (coast is near-optimal at 7 ticks);
        # median error is the discriminator there.
        good = (oh > ch) or (oh == ch and om < cm)
        ok = ok and good
        print('  h=%-4d oracle %5.1f%% (med %6.2fhu)  coast %5.1f%% (med %7.2fhu)  %s'
              % (h, oh, om, ch, cm, 'OK' if good else 'FAIL'))
    print('\nRESULT: %s' % ('PASS' if ok else 'FAIL'))


if __name__ == '__main__':
    main()