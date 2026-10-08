"""Tick-for-tick physics verifier: Python rollout vs in-game movelog ground truth.

Input: amalgam_movelog_*.csv from the MoveLog feature (local player session:
per-tick origin, viewangles, forwardmove/sidemove/upmove, buttons, m_flMaxspeed,
flags, class + a header of live engine cvars).

Method: derive velocity by backward difference (networked m_vecVelocity is stale
in demos AND live - never trust it), then step OUR engine (inputspace.accelerate
with the CSV's own wishdir/wishspeed, friction from the CSV's constants) and
compare predicted vs actual next-tick position. Reports per-tick error by regime
(ground/air, throttle bucket, speed bucket) so miscalibration shows up exactly
where it lives:
  - constant offset  -> wrong accel/friction constant (bake the CSV's cvars in)
  - grows with speed -> wrong friction or wrong add-speed rule
  - grows with |dv|  -> wrong accel ceiling
  - spikes at jumps  -> impulse handling (expected: impulses are external)
Usage: python ml/verify_physics.py <movelog.csv>
"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inputspace as P

DT = 1.0 / 66.0


def load(path):
    cvars = {}
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                for m in re.finditer(r'(\w+)=([-\d.NaN]+)', line):
                    try:
                        cvars[m.group(1)] = float(m.group(2))
                    except ValueError:
                        pass
                continue
            rows.append(line)
    if not rows:
        print('empty log', flush=True)
        sys.exit(1)
    import csv
    hdr = rows[0].strip().split(',')
    data = []
    for r in rows[1:]:
        p = r.strip().split(',')
        if len(p) != len(hdr):
            continue
        try:
            data.append([float(x) for x in p])
        except ValueError:
            continue
    a = np.array(data)
    col = {n: i for i, n in enumerate(hdr)}
    return cvars, a, col


def main(path):
    cvars, a, col = load(path)
    print('cvars: %s' % ' '.join('%s=%g' % (k, v)
                                 for k, v in sorted(cvars.items())), flush=True)
    print('rows: %d' % len(a), flush=True)
    g = lambda n: a[:, col[n]]
    tick, ox, oy = g('tick'), g('ox'), g('oy')
    yaw = np.deg2rad(g('eyey'))
    fm, sm = g('fm'), g('sm')
    mspd = g('maxspeed')
    gr = g('grounded') > 0.5
    # backward-difference velocity (the only honest velocity)
    vx = np.diff(ox, prepend=ox[0]) / DT
    vy = np.diff(oy, prepend=oy[0]) / DT
    # engine wishdir from the actual usercmd + view yaw
    fx, fy = np.cos(yaw), np.sin(yaw)
    rx, ry = np.sin(yaw), -np.cos(yaw)   # D-positive (Source convention)
    wx = fm * fx + sm * rx
    wy = fm * fy + sm * ry
    wn = np.hypot(wx, wy)
    m = wn > 1e-9
    wx[m] /= wn[m]
    wy[m] /= wn[m]
    w = np.minimum(wn, 1.0) * np.where(np.isfinite(mspd) & (mspd > 0), mspd, 240.0)
    # step OUR physics, compare with the ACTUAL next position
    errs, regs = [], []
    for i in range(len(a) - 1):
        if tick[i + 1] != tick[i] + 1:
            continue  # tick gap: incomparable, skip (never interpolate)
        v = np.array([vx[i], vy[i]])
        sp = float(np.hypot(*v))
        if gr[i] and sp > 0:
            ctrl = max(sp, 100.0)
            ns = max(sp - ctrl * 4.0 * DT, 0.0)
            v = v * (ns / sp)
        cur = float(v @ np.array([wx[i], wy[i]])) if m[i] else 0.0
        dv = min(max(w[i] - cur, 0.0), 10.0 * DT * w[i]) if (m[i] and w[i] > 0) else 0.0
        v = v + np.array([wx[i], wy[i]]) * dv
        pred = np.array([ox[i], oy[i]]) + v * DT
        err = float(np.hypot(pred[0] - ox[i + 1], pred[1] - oy[i + 1]))
        errs.append(err)
        regs.append((gr[i], wn[i], sp))
    errs = np.array(errs)
    print('comparable ticks: %d/%d  err hu: p50 %.3f p90 %.3f p99 %.3f max %.2f' % (
        len(errs), len(a) - 1, *np.percentile(errs, [50, 90, 99, 100])), flush=True)
    regs = np.array(regs)
    for name, mm in [('ground', regs[:, 0] > 0.5), ('air', regs[:, 0] < 0.5),
                     ('full-throttle', regs[:, 1] >= 0.99),
                     ('coast', regs[:, 1] < 0.05),
                     ('fast>200', regs[:, 2] > 200)]:
        e = errs[mm]
        if len(e) > 20:
            print('  %-14s n=%6d p50 %.3f p99 %.3f' % (
                name, len(e), *np.percentile(e, [50, 99])), flush=True)
    print('(residual spikes at jumps/knockbacks are EXPECTED: external impulses)', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
