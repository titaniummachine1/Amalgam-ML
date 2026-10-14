"""diag_candidates.py — print all 9 candidate velocities vs the observed one,
so we can SEE why no candidate fits. Ground ticks only, first N.
"""
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (AIRCAP, DUCK_FRAC, DT, FL_DUCKING, FL_ONGROUND, GRAVITY,
                     air_accelerate, ground_step, inverse_ground, load_csv_live,
                     yaw_to_fwd)
from boxtrace import HULL_DUCK, HULL_STAND, load_brushes, move_box
from recover_viterbi import BASE, COAST, KEY_SPEED

csv = sys.argv[1]
brp = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 6
BR = load_brushes(brp)
rows = load_csv_live(csv, classes=set(range(1, 10)))
e = max(rows, key=lambda x: len(rows[x]))
rs = rows[e]
shown = 0
for k in range(200, 4000):
    if shown >= N:
        break
    if k + 1 >= len(rs):
        break
    if int(rs[k + 1]['tick']) - int(rs[k]['tick']) != 1:
        continue
    try:
        p = np.array([float(rs[k]['ox']), float(rs[k]['oy']), float(rs[k]['oz'])])
        v = np.array([float(rs[k]['vx']), float(rs[k]['vy']), float(rs[k]['vz'])])
        vn = np.array([float(rs[k + 1]['vx']), float(rs[k + 1]['vy']),
                       float(rs[k + 1]['vz'])])
        fl = int(float(rs[k]['flags']))
        ey = float(rs[k]['eyey'])
        ms = float(rs[k].get('mspd') or 300.0)
    except Exception:
        continue
    if not (fl & FL_ONGROUND):
        continue
    dk = bool(fl & FL_DUCKING)
    # mspd is already duck-adjusted in the demo CSV; do NOT divide again.
    fwd = yaw_to_fwd(ey)
    right = np.array([fwd[1], -fwd[0]])
    print('--- k=%d yaw=%.1f mspd=%.0f duck=%d' % (k, ey, ms, dk))
    print('    v =[%8.2f %8.2f]  vn=[%8.2f %8.2f]  dv=[%7.3f %7.3f]'
          % (v[0], v[1], vn[0], vn[1], vn[0] - v[0], vn[1] - v[1]))
    print('    fwd=[%6.3f %6.3f] right=[%6.3f %6.3f]' % (fwd[0], fwd[1], right[0], right[1]))
    iv = inverse_ground(v, vn, ms)
    wd = iv['wishdir']
    wds = 'None' if wd is None else '[%6.3f %6.3f]' % (np.asarray(wd).ravel()[0],
                                                       np.asarray(wd).ravel()[1])
    print('    inverse wd=%s ws=%.2f  res=%.5f' % (wds, float(iv['wishspeed']),
                                                  iv['info'].get('res', -1)))
    half = np.array(HULL_DUCK if dk else HULL_STAND)
    for ci in range(9):
        if ci == COAST:
            wd2, w = None, 0.0
        else:
            bx, by = BASE[ci]
            wvel = fwd * (bx * KEY_SPEED) + right * (by * KEY_SPEED)
            mag = float(np.linalg.norm(wvel))
            wd2 = wvel / mag if mag > 1e-9 else None
            w = min(mag, ms)
        out = ground_step(np.array([v[0], v[1], 0.0]),
                          wd2 if wd2 is not None else np.array([1.0, 0.0]), w, ms)
        v2 = out[0] if isinstance(out, tuple) else out
        pend, _h, _n, _u = move_box(p, p + v2 * DT, half, BR)
        pv = (pend - p) / DT
        err = float(np.linalg.norm(np.array([pv[0], pv[1]]) - np.array([vn[0], vn[1]])))
        nm = 'coast' if ci == COAST else '%+d%+d' % (BASE[ci][0], BASE[ci][1])
        print('      c%d %-6s w=%.0f  pred=[%8.2f %8.2f]  err=%8.3f'
              % (ci, nm, w, pv[0], pv[1], err))
    shown += 1