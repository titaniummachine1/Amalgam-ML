"""rollout_gate.py — THE agreement gate: recovered-input rollouts vs truth.
For each clean ground segment: recover per-tick inputs via inverse dynamics
("sane demo inputs"), roll forward through movesim (free-space v1; soup wall
exclusion via ray), compare against factual future positions.
Gate: recovered-input rollout MUST beat frozen-input rollout paired (input
recovery adds information) + absolute caps (med@7<6, med@13<15, med@20<30).
This is also the dataset engine (rollout -> compare -> blame); the gate and
the trainer share this code path by construction.
Segment criteria (clean signal only): 40 consecutive live rows, tick-adjacent
(vgap==1), same class, grounded throughout, per-step dp<50, hp constant,
raw mspd>1, no soup wall within (path length+100) ahead.
Usage: python ml/rollout_gate.py <csv> <soup.npz> [max_seg]
"""
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (AIRCAP, CLASS_SPEED, DUCK_FRAC, FL_DUCKING, FL_ONGROUND,
                     ground_step, inverse_ground, load_csv_live, rollout_free,
                     yaw_to_fwd)
from souptrace import load_soup, ray_hit

HOR = [7, 13, 20, 33]


def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d


def main():
    rows = load_csv_live(sys.argv[1], classes=set(range(1, 10)))
    soup, _, _ = load_soup(sys.argv[2])
    max_seg = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    rec = {h: [] for h in HOR}
    frz = {h: [] for h in HOR}
    nseg = nwall = 0
    for e, rs in rows.items():
        i = 0
        while i + 40 < len(rs) and nseg < max_seg:
            seg = rs[i:i + 41]
            ticks = [int(r['tick']) for r in seg]
            if any(b - a != 1 for a, b in zip(ticks, ticks[1:])):
                i += 1
                continue
            if any(int(f(r['class'])) != int(f(seg[0]['class'])) for r in seg):
                i += 1
                continue
            if any(not (int(f(r['flags'])) & FL_ONGROUND) for r in seg):
                i += 1
                continue
            if any(f(r['mspd']) <= 1 for r in seg):
                i += 1
                continue
            hps = [r.get('health') for r in seg]
            if any(h != hps[0] for h in hps):
                i += 1  # damage taken -> knockback forces, not clean
                continue
            P = np.array([[f(r['ox']), f(r['oy']), f(r['oz'])] for r in seg])
            dp = np.linalg.norm(P[1:] - P[:-1], axis=1)
            if np.any(dp > 50):
                i += 1
                continue
            V = np.array([[f(r['vx']), f(r['vy']), f(r['vz'])] for r in seg])
            cls = int(f(seg[0]['class']))
            ms0 = f(seg[0]['mspd']) or CLASS_SPEED.get(cls, 300.0)
            if int(f(seg[0]['flags'])) & FL_DUCKING:
                ms0 *= DUCK_FRAC
            # wall ahead? ray from chest along initial motion (path+100 margin)
            v0 = V[0]
            sp0 = float(np.hypot(v0[0], v0[1]))
            if sp0 < 50:
                i += 1
                continue
            dd = np.array([v0[0] / sp0, v0[1] / sp0, 0.0])
            plen = float(np.sum(np.linalg.norm(P[1:34] - P[:33], axis=1)))
            o = np.array([P[0][0], P[0][1], P[0][2] + 30.0])
            if ray_hit(o, dd, soup, plen + 100.0) < np.inf:
                nwall += 1
                i += 40
                continue
            # recover inputs per step (demo-derived = sane) + views
            inp = []
            views = []
            ok = True
            for k in range(40):
                ms = ms0
                inv = inverse_ground(V[k], V[k + 1], ms)
                wd = inv['wishdir']
                inp.append((wd if wd is not None else None,
                            inv['wishspeed'] if wd is not None else 0.0))
                try:
                    views.append(yaw_to_fwd(f(seg[k]['eyey'])))
                except ValueError:
                    views.append(None)
            grs = [True] * 40
            pr = rollout_free(V[0], P[0], 40, inp, grs, ms0, AIRCAP, views)
            f0 = inp[0]
            pf = rollout_free(V[0], P[0], 40, [f0] * 40, grs, ms0, AIRCAP, views)
            for h in HOR:
                rec[h].append(float(np.linalg.norm(pr[h - 1] - P[h])))
                frz[h].append(float(np.linalg.norm(pf[h - 1] - P[h])))
            nseg += 1
            i += 40
    print('segments=%d wall-excluded=%d' % (nseg, nwall), flush=True)
    ok = nseg >= 50
    for h in HOR:
        a = np.array(rec[h])
        b = np.array(frz[h])
        med, p90 = (np.median(a), np.percentile(a, 90)) if len(a) else (-1, -1)
        fmed = np.median(b) if len(b) else -1
        print('@%d: recovered med=%.1f p90=%.1f | frozen med=%.1f (n=%d)' % (
            h, med, p90, fmed, len(a)), flush=True)
    caps = {7: 6.0, 13: 15.0, 20: 30.0, 33: 120.0}
    for h in HOR:
        a = np.array(rec[h])
        b = np.array(frz[h])
        if len(a) and (np.median(a) >= np.median(b) or np.median(a) > caps[h]):
            print('GATE=FAIL @%d' % h, flush=True)
            ok = False
    print('GATE=%s' % ('PASS' if ok and nseg >= 50 else 'FAIL'), flush=True)
    sys.exit(0 if ok and nseg >= 50 else 1)


main()
