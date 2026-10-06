"""Unit test for ml/inputspace.py against a SECOND, hand-written implementation of
CBasePlayer::Friction + Accelerate (STATUS.md: always validate against a second
implementation before trusting a headline number).
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inputspace as P

fails = []


def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print('%-46s %14.6f vs %14.6f  %s' % (name, got, want, 'ok' if ok else 'FAIL'))
    if not ok:
        fails.append(name)


# ---- literal transcription of the engine, scalar, float64 -------------------
def ref_step(vx, vy, gr, tau, ang, maxspeed, addspeed=True, rule='dot'):
    wx, wy = tau * math.cos(ang), tau * math.sin(ang)
    wspeed = math.hypot(wx, wy)
    if wspeed > 1e-9:
        wx, wy = wx / wspeed, wy / wspeed
    else:
        wspeed = 0.0
    wspeed = min(wspeed * maxspeed, maxspeed)          # clamped to MaxSpeed
    vmag = math.hypot(vx, vy)
    if gr > 0.5 and vmag > 0.0:
        ctrl = vmag if vmag > 100.0 else 100.0
        new = max(vmag - ctrl * 4.0 * P.DT, 0.0)
        k = new / vmag
        vx, vy, vmag = vx * k, vy * k, new
    if addspeed:
        # CGameMovement::Accelerate projects velocity ONTO THE WISHDIR
        cur = vx * wx + vy * wy if rule == 'dot' else vmag
        if wspeed > 0.0 and (wspeed - cur) > 0.0:
            dv = 10.0 * P.DT * wspeed
            dv = min(dv, wspeed - cur)
            vx += wx * dv
            vy += wy * dv
    else:
        dv = 10.0 * P.DT * wspeed
        vx += wx * dv
        vy += wy * dv
    return vx, vy


CASES = [
    # vx,      vy,   grounded, throttle, angle,  maxspeed, what
    (0.0, 0.0, 1.0, 1.0, 0.0, 240.0, 'stand still, full W'),
    (0.0, 0.0, 1.0, 0.5, 0.0, 240.0, 'half throttle from rest'),
    (200.0, 0.0, 1.0, 1.0, 0.0, 240.0, 'below cap'),
    (240.0, 0.0, 1.0, 1.0, 0.0, 240.0, 'exactly at cap (equilibrium)'),
    (300.0, 0.0, 1.0, 1.0, 0.0, 240.0, 'above cap: add-speed forbids accel'),
    (100.0, 0.0, 1.0, 1.0, 0.7, 240.0, 'angled accel'),
    (100.0, 50.0, 1.0, 0.8, -0.3, 240.0, 'angled, partial'),
    (0.0, 0.0, 0.0, 1.0, 0.0, 240.0, 'airborne from rest'),
    (233.0, 0.0, 0.0, 1.0, 0.0, 240.0, 'airborne fast'),
    (120.0, 120.0, 1.0, 1.0, math.pi / 4, 240.0, '|wishvel| > maxspeed'),
    (5.0, 0.0, 1.0, 1.0, 0.0, 240.0, 'below stop speed'),
    (90.0, 0.0, 1.0, 0.0, 0.0, 240.0, 'no input: friction only'),
    (240.0, 0.0, 1.0, 0.3, 1.1, 240.0, 'at cap, weak off-axis push'),
    (0.0, 240.0, 1.0, 1.0, math.pi / 2, 240.0, 'pure lateral from rest'),
]

for addspeed in (True, False):
    print('\n=== addspeed=%s ===' % addspeed)
    for i, (vx, vy, gr, tau, ang, ms, what) in enumerate(CASES):
        rx, ry = ref_step(vx, vy, gr, tau, ang, ms, addspeed)
        o = torch.tensor([[tau * math.cos(ang), tau * math.sin(ang)]], dtype=torch.float64)
        mag = float(o.norm())
        wh = o / max(mag, 1e-6)
        w = torch.tensor([[ms * min(mag, 1.0)]], dtype=torch.float64)
        v = torch.tensor([[vx, vy]], dtype=torch.float64)
        got = P.accelerate(v, wh, w, torch.tensor([[gr]], dtype=torch.float64),
                           addspeed=addspeed)
        check('case %2d vx  (%s)' % (i, what), float(got[0, 0]), rx)
        check('case %2d vy  (%s)' % (i, what), float(got[0, 1]), ry)

# one-sided: with addspeed ON no POSITIVE accel is possible at/above the cap
# (friction may still decelerate, so assert on the sign, not on zero net change)
print('\n=== engine invariants ===')
o = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
w = torch.tensor([[240.0]], dtype=torch.float64)
for spd in (240.0, 260.0, 400.0):
    v = torch.tensor([[spd, 0.0]], dtype=torch.float64)
    d = float((P.accelerate(v, o, w, torch.tensor([[1.0]]), addspeed=True) - v)[0, 0])
    fric = -max(spd, 100.0) * 4.0 * P.DT
    # at exactly the cap the accel cancels friction (net 0, the engine equilibrium,
    # and the source of the observed grounded speed plateau near 0.94*maxspeed);
    # above the cap only friction acts.
    want = 0.0 if abs(spd - 240.0) < 1e-9 else fric
    check('addspeed on: net delta at |v|=%g' % spd, d, want)
    d2 = float((P.accelerate(v, o, w, torch.tensor([[1.0]]), addspeed=False) - v)[0, 0])
    check('addspeed off: delta == friction + 10*dt*240 at |v|=%g' % spd, d2, fric + 10.0 * P.DT * 240.0)

# friction is the champion's helper, unchanged
v = torch.tensor([[240.0, 0.0]], dtype=torch.float64)
check('friction 240 grounded', float(P.friction(v, torch.tensor([[1.0]]))[0, 0]),
      240.0 - 240.0 * 4.0 * P.DT)
check('friction 240 airborne', float(P.friction(v, torch.tensor([[0.0]]))[0, 0]), 240.0)

print('\n%s' % ('ALL ENGINE CHECKS PASS' if not fails else 'FAILURES: %s' % fails))
sys.exit(1 if fails else 0)
