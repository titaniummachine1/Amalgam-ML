"""gate_wall.py — NO-CHEAT certificate: sim replicates demo wall interactions.
Finds windows where the player runs into a wall (speed>150, wall ahead,
impact slowdown), recovers input packets (inverse_ground, demo flags only
to select ground packets — jumps out of scope), replays them through
WalkMove + hull sweep + down-trace ground (the world of the demo), scores
|predicted - truth| @7/13/20. Pass = sim reacts to environment like engine.
NO future truth enters the rollout (inputs recovered per tick-pair only,
same packets the player sent). Usage:
  python ml/gate_wall.py <csv> <soup.npz> [maxwin]
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (CLASS_SPEED, DUCK_FRAC, FL_DUCKING, FL_ONGROUND,
                      back_clamp, ground_step, friction, inverse_ground,
                      load_csv_live, yaw_to_fwd)
from souptrace import load_soup, build_grid, grid_hit
from sweep import sweep

HIST = 33
FUT = 20
HOR = [7, 13, 20]
BR = None
HS = None
HD = None
MASK_PLAYERSOLID = 0x1 | 0x4000 | 0x10000 | 0x2 | 0x20000 | 0x8  # SOLID|MOVABLE|PCLIP|WINDOW|NPC|GRATE
BR = None
HALF = None


def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d


def grounded_at(p, soup, grid, half=None):
    if BR is not None:
        from boxtrace import grounded_box
        return grounded_box(p, half if half is not None else HS, BR)
    o = np.array([p[0], p[1], p[2] + 1.0])
    t = grid_hit(o, np.array([0.0, 0.0, -1.0]), soup, 4.0, grid)
    return t < 3.0


def main():
    global BR, HS, HD
    csvpath, souppath = sys.argv[1], sys.argv[2]
    maxwin = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    if len(sys.argv) > 4:
        from boxtrace import load_brushes, HULL_STAND, HULL_DUCK
        import os as _os
        drop = []
        dp = _os.path.join(_os.path.dirname(sys.argv[4]), 'brush_drop_beta4b.txt')
        if _os.path.exists(dp):
            drop = [int(l.strip()) for l in open(dp) if l.strip().isdigit()]
        BR = load_brushes(sys.argv[4], drop)
        hxy = float(sys.argv[5]) if len(sys.argv) > 5 else 24.0
        HS = np.array([hxy, hxy, 41.0])
        HD = np.array([hxy, hxy, 31.0])
        print('box path half-xy=%.0f drops=%s (TF hull 24/24/82, duck /62)' % (hxy, drop), flush=True)
    rows = load_csv_live(csvpath, classes=set(range(1, 10)))
    soup, _, _ = load_soup(souppath)
    grid = build_grid(soup)
    errs = {h: [] for h in HOR}
    winfo = []
    nskip_trace = nimpact = 0
    nudges = []
    t0 = time.time()
    for e, rs in rows.items():
        if len(errs[7]) >= maxwin:
            break
        i = 0
        L = HIST + FUT
        while i + L < len(rs) and len(errs[7]) < maxwin:
            win = rs[i:i + L + 1]
            ticks = [int(r['tick']) for r in win]
            if any(b - a != 1 for a, b in zip(ticks, ticks[1:])):
                i += 1
                continue
            if any(int(f(r['class'])) != int(f(win[0]['class'])) for r in win):
                i += 1
                continue
            P = np.array([[f(r['ox']), f(r['oy']), f(r['oz'])] for r in win])
            if np.any(np.linalg.norm(P[1:] - P[:-1], axis=1) > 80):
                i += 1
                continue
            V = np.array([[f(r['vx']), f(r['vy']), f(r['vz'])] for r in win])
            if not all(bool(int(f(r['flags'])) & FL_ONGROUND) for r in win):
                i += 1
                continue
            sp0 = float(np.hypot(V[HIST][0], V[HIST][1]))
            if sp0 < 150:
                i += 1
                continue
            # wall ahead within 250hu?
            dd = np.array([V[HIST][0] / sp0, V[HIST][1] / sp0, 0.0])
            o = np.array([P[HIST][0], P[HIST][1], P[HIST][2] + 30.0])
            if grid_hit(o, dd, soup, 250.0, grid) >= np.inf:
                i += 1
                continue
            # impact: future speed drops (stop or slide along wall)?
            fsp = [float(np.hypot(v[0], v[1])) for v in V[HIST + 1:HIST + FUT + 1]]
            if min(fsp) > 60:
                i += 1
                continue
            nimpact += 1
            # trace-ground must agree with demo flags on every tick (clean gate;
            # per-tick TF hull: duck 62 tall else 82)
            gok = True
            for q in range(HIST, HIST + FUT + 1):
                hh = None
                if BR is not None:
                    hh = HD if bool(int(f(win[q]['flags'])) & FL_DUCKING) else HS
                if not grounded_at(P[q], soup, grid, hh):
                    gok = False
                    break
            if not gok:
                nskip_trace += 1
                i += 1
                continue
            cls = int(f(win[0]['class']))
            ms0 = f(win[0]['mspd']) or CLASS_SPEED.get(cls, 300.0)
            # recover input packets (past pairs only — same packets as demo)
            # + per-tick hull (TF: duck 24/24/62 else 24/24/82, demo flags = replay)
            inp, views, hulls = [], [], []
            ok = True
            for k in range(HIST, HIST + FUT):
                r = win[k]
                try:
                    ey = f(r['eyey'])
                    duck = bool(int(f(r['flags'])) & FL_DUCKING)
                    msk = ms0 * (DUCK_FRAC if duck else 1.0)
                    inv = inverse_ground(V[k], V[k + 1], msk)
                except (ValueError, KeyError, IndexError):
                    ok = False
                    break
                wd = inv['wishdir']
                inp.append((wd, inv['wishspeed'] if wd is not None else 0.0))
                views.append(yaw_to_fwd(ey))
                hulls.append(HD if (BR is not None and duck) else HS)
            if not ok:
                i += 1
                continue
            # WORLD ROLLOUT: physics + sweep + traces, zero future truth
            v = V[HIST].copy()
            p = P[HIST].copy()
            pr = []
            for k in range(FUT):
                wd, ws = inp[k]
                v[2] = 0.0
                if wd is None:
                    v = friction(v)
                else:
                    v, stopped = ground_step(v, wd, min(ws, ms0), ms0)
                    if stopped:
                        v = np.zeros(3)
                v = back_clamp(v, views[k], ms0)
                if BR is not None:
                    from sweep import sweep_box
                    p_new, _, _, ng = sweep_box(p, p + v * 0.015, BR, hulls[k], MASK_PLAYERSOLID)
                    nudges.append(ng)
                else:
                    p_new, _, _ = sweep(p, p + v * 0.015, soup)
                p = p_new
                pr.append(p.copy())
            pr = np.array(pr)
            truth = P[HIST + 1:HIST + FUT + 1]
            for h in HOR:
                errs[h].append(float(np.linalg.norm(pr[h - 1] - truth[h - 1])))
            errs13 = float(np.linalg.norm(pr[12] - truth[12]))
            winfo.append((errs13, int(e), ticks[HIST], P[HIST].round(1).tolist()))
            i += FUT
    for h in HOR:
        a = np.array(errs[h])
        print('wall-gate n=%d err@%d med=%.1f p90=%.1f' % (
            len(a), h, np.median(a) if len(a) else -1, np.percentile(a, 90) if len(a) else -1),
            flush=True)
    print('impact-candidates=%d trace-skipped=%d time=%.0fs' % (nimpact, nskip_trace, time.time() - t0),
          flush=True)
    if nudges:
        print('nudge: windows=%d mean=%.2fhu max=%.2fhu' % (
            sum(1 for x in nudges if x > 0), sum(nudges) / len(nudges), max(nudges)), flush=True)
    for e13, ent, tick, p0 in sorted(winfo, reverse=True)[:4]:
        print('worst ent=%s tick=%d err13=%.1f p0=%s' % (ent, tick, e13, p0), flush=True)


if __name__ == '__main__':
    main()
