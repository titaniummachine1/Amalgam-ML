"""movesim.py — TF2 movement FORMULA port (free-space v1, no collision).

Spec (READ-ONLY reference, never committed, never copied into any repo):
  C:/gitProjects/tf2-source-code
Live files (verified against .vpc build files; game/shared/tf2/ is DEAD code):
  game/shared/tf/tf_gamemovement.cpp : FullWalkMove/WalkMove/AirMove/CheckJumpButton
  game/shared/gamemovement.cpp       : Friction/Accelerate/AirAccelerate/Start+FinishGravity
  game/shared/movevars_shared.cpp    : sv_* defaults
Constants: dt=1/66, friction=4, stop=100, accel=10, airaccel=10, gravity=800,
  aircap=30, jump=289, wish-thresh=40 (=100*4/10), back-clamp 0.9 over 100,
  ground stop 1.0. Duck maxspeed x1/3 and class table are ASSUMPTIONS
  (flagged ASSUME) — validation must confirm/refute them.
v1 scope: formulas only. No traces, no CategorizePosition (caller supplies
  observed grounded flags), no StepMove/slide, no water/grapple/kart/parachute,
  no ForwardPull (attribute, 0 in MGE), base velocity 0.
"""
import math

import numpy as np

DT = 0.015  # TF2 tick_interval (66.67Hz). Confirmed end-to-end: strict-ballistic
# jump arcs fit quad coef -0.0900 == -400*DT^2 exactly (15 arcs, MGE chaos).
FRICTION = 4.0
STOPSPEED = 100.0
ACCEL = 10.0
AIRACCEL = 10.0
GRAVITY = 800.0
AIRCAP = 30.0
JUMP_VEL = 289.0
WISH_THRESH = 100.0 * FRICTION / ACCEL  # = 40
BACK_CLAMP = 0.9
BACK_CLAMP_MIN = 100.0
GROUND_STOP = 1.0
DUCK_FRAC = 1.0 / 3.0  # ASSUME: TF2 duck maxspeed fraction
SF = 1.0  # surface friction (normal ground)

# ASSUME: class max speeds (TF_CLASS order 1..9). Used only when mspd missing.
CLASS_SPEED = {1: 400.0, 2: 300.0, 3: 240.0, 4: 280.0, 5: 320.0,
               6: 230.0, 7: 300.0, 8: 300.0, 9: 300.0}

FL_ONGROUND = 1
FL_DUCKING = 2


def load_csv_live(csvpath, classes=None):
    """Load a demo CSV with gap-aware derived velocity (single source of
    truth for validate/souptrace/dataset builders — never reimplement).
    - STV vel columns are STALE: derive from positions.
    - Extractor repeats rows on ticks without updates (m2f~0.9): liveness =
      pos OR eye OR flags changed vs prev row; stale repeats DROPPED
      (standing-while-aiming survives via eye, v=0 true).
    - v = dp/(gap*DT) with gap = ticks since last live row; pairs need gap==1
      (adjacency) — enforced by callers via r['vgap'].
    Returns {ent: [rows]} with vx/vy/vz/vgap set. Rows: tick,ent,class,ox..vz,
    eyep,eyey,mspd,flags,health,life,team,fresh + vx,vy,vz,vgap.
    """
    import csv as _csv
    from collections import defaultdict as _dd
    rows = _dd(list)
    with open(csvpath, newline='') as fh:
        for r in _csv.DictReader(fh):
            try:
                if classes is None or int(float(r['class'])) in classes:
                    rows[int(r['ent'])].append(r)
            except ValueError:
                continue
    for e in rows:
        rows[e].sort(key=lambda r: int(r['tick']))
        rs = rows[e]
        live = []
        prev = None
        for r in rs:
            if prev is None:
                live.append(r)
            else:
                try:
                    changed = (r['ox'] != prev['ox'] or r['oy'] != prev['oy'] or
                               r['oz'] != prev['oz'] or r['eyey'] != prev['eyey'] or
                               r['flags'] != prev['flags'])
                except KeyError:
                    changed = True
                if changed:
                    live.append(r)
            prev = r
        for a, b in zip(live, live[1:]):
            try:
                gap = int(b['tick']) - int(a['tick'])
                if gap < 1:
                    continue
                k = 1.0 / (gap * DT)
                a['vx'] = (float(b['ox']) - float(a['ox'])) * k
                a['vy'] = (float(b['oy']) - float(a['oy'])) * k
                a['vz'] = (float(b['oz']) - float(a['oz'])) * k
                a['vgap'] = gap
            except ValueError:
                pass
        rows[e] = [r for r in live if 'vgap' in r]
    return rows


def friction(v, sf=SF):
    """Base Friction (gamemovement.cpp:1612). Scales full vector."""
    v = np.asarray(v, dtype=float).copy()
    speed = float(np.linalg.norm(v))
    if speed < 0.1:
        return v
    control = STOPSPEED if speed < STOPSPEED else speed  # TF_DLL branch
    drop = control * FRICTION * sf * DT
    newspeed = max(speed - drop, 0.0)
    if newspeed != speed:
        v *= newspeed / speed
    return v


def accelerate(v, wishdir, wishspeed, accel=ACCEL, sf=SF):
    """Base Accelerate (gamemovement.cpp:1822). DOT-based current speed."""
    v = np.asarray(v, dtype=float).copy()
    wishdir = np.asarray(wishdir, dtype=float)
    cur = float(np.dot(v, wishdir))
    add = wishspeed - cur
    if add <= 0:
        return v
    acc = accel * DT * wishspeed * sf
    if acc > add:
        acc = add
    v += acc * wishdir
    return v


def air_accelerate(v, wishdir, wishspeed, cap=AIRCAP, accel=AIRACCEL, sf=SF):
    """Base AirAccelerate (gamemovement.cpp:1707)."""
    v = np.asarray(v, dtype=float).copy()
    wishdir = np.asarray(wishdir, dtype=float)
    wc = wishspeed if wishspeed <= cap else cap
    cur = float(np.dot(v, wishdir))
    add = wc - cur
    if add <= 0:
        return v
    acc = accel * wishspeed * DT * sf
    if acc > add:
        acc = add
    v += acc * wishdir
    return v


def ground_step(v, fwd, wishspeed, maxspeed, sf=SF):
    """Live WalkMove velocity update (tf_gamemovement.cpp:1747), no collision.
    fwd = 2D unit wishdir. Returns (v_new, stopped)."""
    v = np.array([v[0], v[1], 0.0], dtype=float)
    v = friction(v, sf)
    acc = ACCEL
    if 0.0 < wishspeed < WISH_THRESH:
        sp = float(np.linalg.norm(v))
        control = STOPSPEED if sp < STOPSPEED else sp
        acc = (control * FRICTION) / wishspeed + 1.0
    v = accelerate(v, np.array([fwd[0], fwd[1], 0.0]), wishspeed, acc, sf)
    sp2 = float(np.hypot(v[0], v[1]))
    if sp2 > maxspeed:  # post-accel 2D clamp (1793)
        v[0] *= maxspeed / sp2
        v[1] *= maxspeed / sp2
    # back-speed clamp 0.9 (1816), needs view forward: applied by caller via
    # back_clamp() since fwd here is wishdir, not view. See below.
    if float(np.linalg.norm(v)) < GROUND_STOP:
        return np.zeros(3), True
    return v, False


def back_clamp(v, view_fwd, maxspeed):
    """Live back-speed clamp (tf_gamemovement.cpp:1816). view_fwd = 2D unit."""
    v = np.asarray(v, dtype=float).copy()
    if BACK_CLAMP >= 1.0 or float(np.linalg.norm(v)) <= BACK_CLAMP_MIN:
        return v
    if float(view_fwd[0] * v[0] + view_fwd[1] * v[1]) < 0:
        bx, by = view_fwd[0], view_fwd[1]
        dot = bx * v[0] + by * v[1]
        backx, backy = bx * dot, by * dot
        bs = float(math.hypot(backx, backy))
        maxb = maxspeed * BACK_CLAMP
        if bs > maxb:
            backx *= maxb / bs
            backy *= maxb / bs
        # reassemble: back + right components
        rx, ry = -by, bx
        rdot = rx * v[0] + ry * v[1]
        v[0] = backx + rx * rdot
        v[1] = backy + ry * rdot
        sp = float(np.linalg.norm(v))
        if sp > maxspeed:
            v[0] *= maxspeed / sp
            v[1] *= maxspeed / sp
    return v


def yaw_to_fwd(eyeyaw_deg):
    r = math.radians(eyeyaw_deg)
    return np.array([math.cos(r), math.sin(r)], dtype=float)


def inverse_ground(v0, v1, maxspeed, sf=SF):
    """Recover (wishdir2, wishspeed, info) from an observed ground tick pair.
    Returns dict; wishdir None when no input. info holds capped/boosted flags
    and the forward-sim residual (implementation-consistency check)."""
    v0 = np.array([v0[0], v0[1], 0.0], dtype=float)
    v1 = np.array([v1[0], v1[1], 0.0], dtype=float)
    vf = friction(v0, sf)
    dv = v1 - vf
    n = float(np.linalg.norm(dv))
    info = {'capped': False, 'boosted': False, 'clamped2d': False, 'res': 0.0}
    if n < 1e-6:
        return {'wishdir': None, 'wishspeed': 0.0, 'info': info}
    wd = dv / n
    cur = float(np.dot(vf, wd))
    # try uncapped, unboosted solution
    ws = n / (ACCEL * DT * sf)
    if ws >= WISH_THRESH and ws - cur >= n - 1e-6:
        wsv, _ = ground_step(v0, wd, ws, 1e9, sf)  # no 2D clamp for residual
        info['res'] = float(np.linalg.norm(wsv - v1))
        if float(np.hypot(v1[0], v1[1])) >= maxspeed - 1.0:
            info['clamped2d'] = True
        return {'wishdir': wd, 'wishspeed': ws, 'info': info}
    # boosted regime (0 < ws < 40): bisect ws s.t. accel formula yields n
    def _mag(w):
        sp = float(np.linalg.norm(vf))
        control = STOPSPEED if sp < STOPSPEED else sp
        a = (control * FRICTION) / w + 1.0
        acc = min(a * DT * w * sf, w - cur)
        return max(acc, 0.0)
    lo, hi = 1e-3, WISH_THRESH
    if _mag(hi) < n - 1e-6:
        # capped regime: wishspeed lower bound
        info['capped'] = True
        return {'wishdir': wd, 'wishspeed': cur + n, 'info': info}
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _mag(mid) < n:
            lo = mid
        else:
            hi = mid
    info['boosted'] = True
    return {'wishdir': wd, 'wishspeed': 0.5 * (lo + hi), 'info': info}


def inverse_air(v0, v1, cap=AIRCAP, sf=SF):
    """Recover air input from observed pair. Checks vz against -g*dt
    (Start+Finish halves sum to a full step)."""
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    dv = v1 - v0
    dvz_clean = abs(dv[2] + GRAVITY * DT) < 25.0
    info = {'dvz_clean': dvz_clean, 'capped': False, 'res': 0.0}
    dxy = dv[:2]
    n = float(np.linalg.norm(dxy))
    if n < 1e-6:
        return {'wishdir': None, 'wishspeed': 0.0, 'info': info}
    wd = dxy / n
    cur = float(v0[0] * wd[0] + v0[1] * wd[1])
    ws = n / (AIRACCEL * DT * sf)
    if ws <= cap and ws - cur >= n - 1e-6:
        return {'wishdir': wd, 'wishspeed': ws, 'info': info}
    info['capped'] = True
    return {'wishdir': wd, 'wishspeed': max(ws, cur + n), 'info': info}


def rollout_free(v0, p0, steps, inputs, grounded, maxspeed, cap=AIRCAP, view_fwds=None):
    """Free-space rollout (no collision): inputs = list of (wishdir2|None,
    wishspeed) per step; grounded = list of bool. view_fwds = optional list
    of 2D unit view-forward per step (enables the live back-speed clamp).
    Returns positions array."""
    v = np.asarray(v0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    out = []
    for k in range(steps):
        wd, ws = inputs[k]
        if grounded[k]:
            v[2] = 0.0
            if wd is None:
                v = friction(v)
            else:
                v, stopped = ground_step(v, wd, min(ws, maxspeed), maxspeed)
                if stopped:
                    v = np.zeros(3)
            if view_fwds is not None and view_fwds[k] is not None:
                v = back_clamp(v, view_fwds[k], maxspeed)
            p = p + v * DT
        else:
            v[2] -= GRAVITY * 0.5 * DT  # StartGravity half
            if wd is not None:
                v = air_accelerate(v, np.array([wd[0], wd[1], 0.0]), ws, cap)
            p = p + v * DT
            v[2] -= GRAVITY * 0.5 * DT  # FinishGravity half
        out.append(p.copy())
    return np.array(out)
