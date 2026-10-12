"""sweep.py — gate-grade swept-box move for offline rollouts (no game).
Approximates CTFGameMovement::TryPlayerMove using soup raycasts as the trace
primitive. Documented approximations vs engine: (1) 9-ray hull sample instead
of true swept box (corners +/-20xy at heights 4/36/68; standing hull
24/72); (2) single slide iteration, no 4-bump loop; (3) step-up = single
raise-sweep-lower attempt (18hu); (4) no entity collision (players/props
handled separately); (5) surface friction/gravity owned by caller (movesim).
Use: sim rollouts that must stop/slide at walls (dataset v2, full-sim gate).
Validation: swept rollout on open segments == free-space rollout (no false
collisions); wall-approach segments stop at truth (swept gate, next).
Spec: C:/gitProjects/tf2-source-code/game/shared/tf/tf_gamemovement.cpp
TryPlayerMove + ClipVelocity (read-only ref, never vendored).
"""
import numpy as np

import sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from souptrace import ray_hit

STEP = 18.0
STOP_EPS = 0.1
HULL_XY = 20.0
HEIGHTS = (4.0, 36.0, 68.0)


def clip_velocity(v, n, overbounce=1.0):
    """Engine ClipVelocity (slide): remove normal component, epsilon-stop."""
    v = np.asarray(v, dtype=float)
    n = np.asarray(n, dtype=float)
    backoff = float(np.dot(v, n)) * overbounce
    out = v - n * backoff
    out[np.abs(out) < STOP_EPS] = 0.0
    return out


def _sweep_once(p0, p1, soup):
    """Move p0->p1 with collision. Returns (pend, hit, normal|None).
    hit = True if any sample ray hit within segment+1hu."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return p1.copy(), False, None, np.zeros(3)
    dn = d / dist
    best = np.inf
    best_hit = None
    for dx in (-HULL_XY, 0.0, HULL_XY):
        for dy in (-HULL_XY, 0.0, HULL_XY):
            for h in HEIGHTS:
                o = p0 + np.array([dx, dy, h])
                t = ray_hit(o, dn, soup, dist + 80.0)
                if t < best:
                    best = t
                    best_hit = o + dn * t
    if best >= np.inf:
        return p1.copy(), False, None, np.zeros(3)
    # segment fraction of the hit point (project onto move dir), backed off
    # 0.5hu so the hull rests on the surface instead of inside it.
    # Contact beyond this move (frac>=1) = no block for this step.
    frac = (float(np.dot(best_hit - p0, dn)) - 0.5) / dist
    if frac >= 1.0:
        return p1.copy(), False, None, np.zeros(3)
    frac = max(frac, 0.0)
    p_hit = p0 + d * frac
    remain = d * (1.0 - frac)
    # hit normal: nearest tri to p_hit (brute force, gate-scale ok)
    d2 = ((soup - p_hit) ** 2).sum(axis=(1, 2))
    tri = soup[int(np.argmin(d2))]
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    nl = float(np.linalg.norm(n))
    n = n / nl if nl > 1e-9 else np.array([0.0, 0.0, 1.0])
    if float(np.dot(n, dn)) > 0:
        n = -n
    slid = clip_velocity(remain, n)
    # step-up attempt: raise 18, re-sweep remainder, drop back down
    if float(np.linalg.norm(slid)) < float(np.linalg.norm(remain)) * 0.3:
        up = p_hit + np.array([0.0, 0.0, STEP])
        r2 = remain.copy()
        d2n = float(np.linalg.norm(r2))
        if d2n > 1e-9:
            t2 = np.inf
            o2 = up + np.array([0.0, 0.0, -STEP + 4.0])
            t2 = ray_hit(o2, r2 / d2n, soup, d2n + 1.0)
            if t2 >= d2n:
                return up + r2 - np.array([0.0, 0.0, STEP]), True, n, np.zeros(3)
    # NOTE: return the CONTACT point (not contact+slide): the bump loop in
    # sweep() re-sweeps the slid remainder, so multi-contact paths (wall
    # graze then floor) resolve correctly instead of tunneling past.
    return p_hit, True, n, slid


def sweep(p0, p1, soup, max_bumps=4):
    """Engine-style bump loop: re-sweep the slid remainder after each contact
    (up to max_bumps), so a wall graze can't skip a floor below.
    Returns (pend, hit_any, last_normal|None). Step-up is attempted inside
    each bump by _sweep_once when motion is heavily blocked."""
    p0 = np.asarray(p0, dtype=float)
    rem = np.asarray(p1, dtype=float) - p0
    p_cur = p0.copy()
    hit_any = False
    normal = None
    for _ in range(max_bumps):
        if float(np.linalg.norm(rem)) < 1e-6:
            break
        p_end, hit, n, rem = _sweep_once(p_cur, p_cur + rem, soup)
        p_cur = p_end
        if hit:
            hit_any = True
            normal = n
        else:
            break
    return p_cur, hit_any, normal


def selftest(soup):
    """Synthetic box + soup floor. Raises on failure."""
    box = np.array([[[0., 0., 0.], [10., 0., 0.], [0., 10., 0.],
                     ]], dtype=float).reshape(1, 3, 3)
    p, hit, _ = sweep(np.array([2., 2., 5.]), np.array([2., 2., -5.]), box)
    assert hit and abs(p[2]) < 3.0, ('floor', p)
    p, hit, _ = sweep(np.array([2., 2., 5.]), np.array([2., 12., 5.]), box)
    assert not hit, ('open', p)
    # soup floor under demo point (regression: must hit near z=49)
    p, hit, _ = sweep(np.array([5866., 6256., 120.]), np.array([5866., 6256., -200.]), soup)
    assert hit and 30.0 < p[2] < 70.0, ('soup-floor', p)
    # open horizontal segment (free-path regression: must not hit)
    p, hit, _ = sweep(np.array([5866., 6256., 80.]), np.array([5966., 6256., 80.]), soup)
    assert not hit, ('free-path', p)
    return True
