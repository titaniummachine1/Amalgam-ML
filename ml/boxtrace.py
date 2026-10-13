"""boxtrace.py — real swept-box trace against brush planes (my own code).
Semantics mirror the engine trace (same math order, same outputs) without
vendoring anything: swept AABB vs convex brush half-spaces by the slab
method; planes offset by support |n|.half (box corners handled exactly,
not sampled). Contents mask filters (PLAYERSOLID etc).
Player hull half extents are CALIBRATED from demos (wall-gate), not copied.
API:
  load_brushes(path) -> dict(bplanes (B,S,4), bnsides, bcontents, bmins, bmaxs)
  trace_box(p0, p1, half, BR, mask) -> (frac, normal|None, contents, startsolid)
    frac in [0,1] (1 = no hit); normal = outward enter-plane normal.
"""
import numpy as np

import sys
sys.path.insert(0, __import__('os').path.dirname(__file__))


def _default_drop(path):
    """Auto-load `<map>.drop` next to the sidecar. The verified phantom list
    (31 ids:7 giant orphans + 24 grazers) must ALWAYS be excluded — a giant
    like6419 encloses every player by hundreds of units and pins the swept box
    in place, which silently destroys all collision (measured: 3hu/tick drag)."""
    import os as _os
    cand = _os.path.splitext(path)[0] + '.drop'
    if not _os.path.exists(cand):
        base = _os.path.basename(_os.path.splitext(path)[0])
        base = base[6:] if base.startswith('brush') else base
        cand = _os.path.join(_os.path.dirname(path), 'brush_drop_%s.txt' % base.replace('brush', ''))
    if _os.path.exists(cand):
        with open(cand) as fh:
            return set(int(x) for x in fh.read().split())
    return set()


def load_brushes(path, drop=None):
    d = np.load(path, allow_pickle=True)
    n = len(d['bnsides'])
    if drop is None:
        drop = _default_drop(path)
    drop = set(drop)
    keep = np.array([i for i in range(n) if i not in drop], dtype=np.int64)
    return dict(bplanes=d['bplanes'][keep].astype(np.float64),
                bnsides=d['bnsides'][keep].astype(np.int64),
                bcontents=d['bcontents'][keep].astype(np.uint32),
                bmins=d['bmins'][keep].astype(np.float64),
                bmaxs=d['bmaxs'][keep].astype(np.float64))


CONT_EPS = 0.05   # contact epsilon: exact touching is not penetration


def trace_box(p0, p1, half, BR, mask=0xFFFFFFFF):
    """Swept box p0->p1 (center path), half extents xyz. Returns
    (frac, normal|None, contents, startsolid). No hit -> (1.0, None, 0, False).
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    half = np.asarray(half, dtype=float)
    d = p1 - p0
    seg_min = np.minimum(p0, p1) - half - 1.0
    seg_max = np.maximum(p0, p1) + half + 1.0
    BP, BN, BC = BR['bplanes'], BR['bnsides'], BR['bcontents']
    ov = np.all(BR['bmaxs'] >= seg_min, axis=1) & np.all(BR['bmins'] <= seg_max, axis=1)
    ov &= (BC & mask) != 0
    cand = np.nonzero(ov)[0]
    best_frac = 1.0
    best_n = None
    best_c = 0
    startsolid = False
    for bi in cand:
        nside = int(BN[bi])
        if nside == 0:
            continue
        P = BP[bi, :nside]
        N = P[:, :3]
        # offset planes by box support
        off = P[:, 3] + np.abs(N) @ half
        nd = N @ p0 - off          # >0 means outside this plane
        den = N @ d
        # Inside all half-spaces => embedded. Exact touching (nd == 0) is NOT
        # solid: a player standing on the floor has the hull bottom face exactly
        # on the surface, and treating that as startsolid made move_box unstick
        # every tick (~3hu/tick of injected error). Engine traces with a hull
        # shrunk by a fraction of a unit for the same reason.
        if np.all(nd <= -CONT_EPS):
            startsolid = True
            best_frac = 0.0
            best_n = N[int(np.argmax(nd))]
            best_c = int(BC[bi])
            break
        # planes already violated (outside) with den>=0 can never enter
        if np.any((nd > 0) & (den >= 0)):
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -nd / np.where(den != 0, den, 1.0)
        # entering planes: outside now (nd>0), moving in (den<0)
        entering = (nd > 0) & (den < 0)
        if not np.any(entering):
            continue
        t_enter = float(np.max(t[entering]))
        # leaving planes must not close before we enter
        leaving = den > 0
        t_leave = float(np.min(t[leaving])) if np.any(leaving) else np.inf
        if t_enter < t_leave and t_enter < best_frac and t_enter >= 0.0:
            best_frac = t_enter
            ei = int(np.argmax(np.where(entering, t, -np.inf)))
            best_n = N[ei].copy()
            best_c = int(BC[bi])
    if best_frac >= 1.0:
        return 1.0, None, 0, startsolid
    return best_frac, best_n, best_c, startsolid


# TF hulls (tf_gamerules.cpp: stand (-24,-24,0)-(24,24,82), duck
# (-24,-24,0)-(24,24,62); bottom exactly at abs origin). Box center =
# origin + up*half[2]. Ground check traces the FULL hull from the origin
# down 2hu (tf_gamemovement.cpp CategorizePosition order: after the move).
HULL_STAND = (24.0, 24.0, 41.0)
HULL_DUCK = (24.0, 24.0, 31.0)
GROUND_DROP = 2.0


def clip_velocity(v, n, overbounce=1.0):
    """Slide: remove normal component, epsilon-stop (same as sweep.py)."""
    v = np.asarray(v, dtype=float)
    n = np.asarray(n, dtype=float)
    backoff = float(np.dot(v, n)) * overbounce
    out = v - n * backoff
    out[np.abs(out) < 0.1] = 0.0
    return out


def unstick(c, half, BR, mask=0xFFFFFFFF, max_pen=2.0, iters=3):
    """Replay-only unstick with full accounting. Truth starts sub-hu embedded
    (STV rounding; tri soup is EPS-expanded so rays never noticed). Finds the
    SHALLOWEST embedding face among all overlapping brushes and nudges out
    along it. Only nudges pen < max_pen (deep stucks stay stuck — real).
    Returns (c_new, nudged_total)."""
    c = np.asarray(c, dtype=float)
    nudged = 0.0
    BP, BN, BC = BR['bplanes'], BR['bnsides'], BR['bcontents']
    for _ in range(iters):
        seg_min = c - half - 1.0
        seg_max = c + half + 1.0
        ov = np.nonzero(np.all(BR['bmaxs'] >= seg_min, axis=1)
                        & np.all(BR['bmins'] <= seg_max, axis=1)
                        & ((BC & mask) != 0))[0]
        best_pen, best_n = np.inf, None
        for bi in ov:
            ns = int(BN[bi])
            if ns == 0:
                continue
            P = BP[bi, :ns]
            off = P[:, 3] + np.abs(P[:, :3]) @ half
            nd = P[:, :3] @ c - off
            if np.all(nd <= 0):
                j = int(np.argmax(nd))
                pen = float(-nd[j])
                if pen < best_pen:
                    best_pen, best_n = pen, P[j, :3].copy()
        if best_n is None or best_pen >= max_pen:
            break
        c = c + best_n * (best_pen + 0.5)
        nudged += best_pen + 0.5
    return c, nudged


def move_box(p0, p1, half, BR, mask=0xFFFFFFFF, max_bumps=4, step=18.0):
    """Bump-loop move of feet-origin p (box center = p + up*half[2]).
    Same application order as the engine move: trace -> contact -> slide
    remainder (ClipVelocity) -> step-up attempt when heavily blocked ->
    re-trace slid remainder. Returns (pend, hit_any, last_normal|None, nudged).
    UNSTICK (replay-only, reported): truth starts sub-hu embedded (STV
    rounding; the live game never re-traces from truth). If the start is
    inside solid, nudge out along the reported normal (<=3 x 0.5hu) before
    the move; nudged distance is returned so gates can report it."""
    p0 = np.asarray(p0, dtype=float)
    up = np.array([0.0, 0.0, half[2]])
    # Collision hull is shrunk slightly on the vertical axis. A player standing
    # on the floor has its origin ON the surface, so the exact hull overlaps the
    # floor brush by a fraction of a unit every tick; with no shrink, every
    # tick reported startsolid and the unstick shoved the box 1.5hu sideways
    # (pinned players, ~3hu/tick error). The engine's own hull is likewise
    # slightly smaller than the collision bounds it reports.
    chalf = np.asarray(half, dtype=float).copy()
    chalf[2] = max(chalf[2] - CONT_EPS, 1.0)
    c0 = p0 + np.array([0.0, 0.0, chalf[2]])
    rem = np.asarray(p1, dtype=float) - p0
    c_cur = c0.copy()
    nudged = 0.0
    for _ in range(3):
        f0, n0, _, ss0 = trace_box(c_cur, c_cur, chalf, BR, mask)
        if not ss0 or n0 is None:
            break
        c_cur = c_cur + n0 * 0.5
        nudged += 0.5
    hit_any = False
    normal = None
    for _ in range(max_bumps):
        if float(np.linalg.norm(rem)) < 1e-6:
            break
        f, n, _, _ = trace_box(c_cur, c_cur + rem, chalf, BR, mask)
        if f >= 1.0:
            c_cur = c_cur + rem
            break
        hit_any = True
        normal = n
        c_cur = c_cur + rem * max(f - 1e-4, 0.0)
        remain = rem * (1.0 - f)
        slid = clip_velocity(remain, n)
        # step-up attempt when heavily blocked
        if float(np.linalg.norm(slid)) < float(np.linalg.norm(remain)) * 0.3:
            up_c = c_cur + np.array([0.0, 0.0, step])
            f2, _, _, _ = trace_box(up_c, up_c + remain, chalf, BR, mask)
            if f2 >= 1.0:
                c_cur = up_c + remain
                rem = np.array([0.0, 0.0, -step])
                continue
        rem = slid
    out = c_cur - np.array([0.0, 0.0, chalf[2]])
    if nudged > 0.0:
        # Unstick is a replay aid for STV-rounding penetration only; it must
        # never become motion. Drop the lateral component of the correction.
        out[0] -= nudged * float(normal[0]) if normal is not None else 0.0
        out[1] -= nudged * float(normal[1]) if normal is not None else 0.0
    return out, hit_any, normal, nudged


def grounded_box(p, half, BR, mask=0xFFFFFFFF, drop=GROUND_DROP):
    """Full-hull trace from the origin down `drop` hu (engine: origin to
    origin-2 with VEC hull). Grounded iff the hull touches within the drop."""
    p = np.asarray(p, dtype=float)
    f, _, _, _ = trace_box(p, p - np.array([0.0, 0.0, drop]), half, BR, mask)
    return f < 1.0


def selftest():
    """One box brush (10,10,10)-(20,20,40); slam a 24-wide box into it."""
    P = np.array([[1, 0, 0, 20], [-1, 0, 0, -10], [0, 1, 0, 20],
                  [0, -1, 0, -10], [0, 0, 1, 40], [0, 0, -1, -10]], dtype=float)
    S = len(P)
    BR = dict(bplanes=np.zeros((1, S, 4)), bnsides=np.array([S]),
              bcontents=np.array([1], dtype=np.uint32),
              bmins=np.array([[10., 10., 10.]]), bmaxs=np.array([[20., 20., 40.]]))
    BR['bplanes'][0] = P
    half = np.array([12.0, 12.0, 36.0])
    f, n, c, ss = trace_box(np.array([0., 15., 0.]), np.array([30., 15., 0.]), half, BR, 1)
    # box front face (center+12) travels 0->30; wall plane x=10 -> contact at
    # center = 10-12 = -2?? starts left of wall: front at 12 > 10?! start inside
    # x-slab (center 0 within [10-12, 20+12]=[-2,32]) but outside y? y=15 in
    # [10-12,20+12] and z=0 in [10-36,40+36] -> startsolid x only... all three
    # slabs contain -> startsolid. Adjust: start at x=-10.
    f, n, c, ss = trace_box(np.array([-10., 15., 0.]), np.array([30., 15., 0.]), half, BR, 1)
    contact_x = -10 + f * 40.0
    print('frac=%.4f contact_x=%.2f (expect -2.00) normal=%s startsolid=%s' % (
        f, contact_x, n, ss))
    assert abs(contact_x - (-2.0)) < 0.05, 'box contact wrong'
    assert n is not None and abs(n[0] + 1.0) < 1e-9 and abs(n[1]) < 1e-9, 'normal wrong'
    f2, _, _, _ = trace_box(np.array([-10., 15., 0.]), np.array([-5., 15., 0.]), half, BR, 1)
    assert f2 >= 1.0, 'false positive'
    print('boxtrace selftest PASS', flush=True)


if __name__ == '__main__':
    selftest()
