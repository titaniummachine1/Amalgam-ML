"""bisect_recover.py — is the failure in tick_step or in the candidate set?
Feeds the INVERTED input (inverse_ground/inverse_air, which internally
forward-simulates to residual 0.0000) through recover_keys.tick_step and
measures the residual. This must be ~0 or tick_step is wrong. If it is ~0 but
the best-of-9 residual is large, the candidate set/speed model is wrong.
"""
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (DUCK_FRAC, FL_DUCKING, FL_ONGROUND, inverse_air,
                     inverse_ground, load_csv_live, yaw_to_fwd)
from boxtrace import load_brushes
from recover_keys import BASE, tick_step

csv = sys.argv[1]
brp = sys.argv[2]
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 400
BR = load_brushes(brp)
rows = load_csv_live(csv, classes=set(range(1, 10)))
e = max(rows, key=lambda x: len(rows[x]))
rs = rows[e]
inv_res, best9_res, true_in_set = [], [], []
n = 0
for k in range(len(rs) - 1):
    if n >= NT:
        break
    try:
        if int(rs[k + 1]['tick']) - int(rs[k]['tick']) != 1:
            continue
        p = np.array([float(rs[k]['ox']), float(rs[k]['oy']), float(rs[k]['oz'])])
        v = np.array([float(rs[k]['vx']), float(rs[k]['vy']), float(rs[k]['vz'])])
        vn = np.array([float(rs[k + 1]['vx']), float(rs[k + 1]['vy']), float(rs[k + 1]['vz'])])
        fl = int(float(rs[k]['flags']))
        ey = float(rs[k]['eyey'])
        mspd = float(rs[k].get('mspd') or 300.0)
    except (ValueError, KeyError, IndexError):
        continue
    duck = bool(fl & FL_DUCKING)
    gr = bool(fl & FL_ONGROUND)
    ms = mspd if mspd > 1.0 else 300.0
    if duck:
        ms *= DUCK_FRAC
    fwd = yaw_to_fwd(ey)
    try:
        iv = inverse_ground(v, vn, ms) if gr else inverse_air(v, vn)
    except Exception:
        continue
    wd = iv['wishdir']
    ws = float(iv['wishspeed'])
    wd2 = None if wd is None else np.asarray(wd, dtype=float).ravel()[:2]
    _p, pv = tick_step(p, v, wd2, ws, ms, gr, duck, BR, fwd)
    inv_res.append(float(np.linalg.norm(pv - vn)))
    # best of the 9 discrete candidates
    best = 1e18
    for bx, by in list(BASE) + [(0, 0)]:
        if bx == 0 and by == 0:
            cw, cws = None, 0.0
        else:
            cw = bx * fwd + by * np.array([fwd[1], -fwd[0]])
            nw = float(np.linalg.norm(cw))
            cw = cw / nw if nw > 1e-9 else None
            cws = ms
        _q, qv = tick_step(p, v, cw, cws, ms, gr, duck, BR, fwd)
        best = min(best, float(np.linalg.norm(qv - vn)))
    best9_res.append(best)
    if wd2 is not None:
        nrm = max(float(np.linalg.norm(wd2)), 1e-9)
        idx = [np.array([bx, by], float) for bx, by in BASE]
        d = [abs(float(np.dot(wd2 / nrm, i / np.linalg.norm(i))) - 1.0) for i in idx]
        true_in_set.append(float(np.min(d)))
    n += 1

I = np.array(inv_res)
B = np.array(best9_res)
print('ticks=%d' % n)
print('INVERTED input through tick_step : med %.4f p90 %.4f   <- must be ~0' %
      (np.median(I), np.percentile(I, 90)))
print('BEST-of-9 candidate residual    : med %.4f p90 %.4f' %
      (np.median(B), np.percentile(B, 90)))
if true_in_set:
    T = np.array(true_in_set)
    print('true-dir nearest of 8, ang err  : med %.5f rad  p90 %.5f  (<0.05=clean)' %
          (np.median(T), np.percentile(T, 90)))
    print('frac ticks where true dir within 10deg of a compass dir: %.3f' %
          (T < 0.05).mean())