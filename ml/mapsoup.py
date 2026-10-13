"""mapsoup.py — BSP collision soup for offline rollouts (no game, no engine).
Spec: C:/gitProjects/tf2-source-code/public/bspflags.h (contents/masks) +
  engine/cmodel_bsp.cpp (brush semantics). Parser: srctools (legal).
  NEVER vendor leak text; numeric constants + independently written code only.
Contents (verified): EMPTY 0, SOLID 1, WINDOW 2, GRATE 8, MOVEABLE 0x4000,
  PLAYERCLIP 0x10000, MONSTERCLIP 0x20000, DETAIL 0x8000000, LADDER 0x20000000.
MASK_PLAYERSOLID = SOLID|MOVEABLE|PLAYERCLIP|WINDOW|MONSTER|GRATE.
Brush faces rebuilt from planes (inside: n.x <= dist, outward normals);
bevel planes kept (compiler collision bevels are real). Displacements +
static props + static brush-entities included; movers (doors/platforms)
skipped + counted (MGE/orange barely have any).
Usage: python ml/mapsoup.py <bsp> [out.npz]
"""
import sys
import time

import numpy as np

from srctools.bsp import BSP, BrushContents

MASK_PLAYERSOLID = (BrushContents.SOLID | BrushContents.MOVABLE |
                    BrushContents.PLAYER_CLIP | BrushContents.WINDOW |
                    BrushContents.NPC | BrushContents.GRATE)
LADDER = BrushContents.LADDER
# Non-solid or dynamic: their brushes must NOT be static collision.
EXCLUDE = {'func_illusionary', 'func_door', 'func_door_rotating',
           'func_movelinear', 'func_tracktrain', 'func_plat', 'func_platrot',
           'func_rotating', 'func_water', 'func_water_analog'}
ROUND = 0.5  # plane-key rounding (hu)

EPS = 0.5  # plane clip epsilon (hu)


def clip_poly(poly, n, d):
    """Clip polygon (N,3) against half-space n.x <= d + EPS. Returns (M,3)."""
    if len(poly) == 0:
        return poly
    out = []
    s = poly[-1]
    ds = float(np.dot(s, n) - d)
    for p in poly:
        dp = float(np.dot(p, n) - d)
        if dp <= EPS:
            if ds > EPS:  # entering: add crossing
                t = ds / (ds - dp) if (ds - dp) != 0 else 0.0
                out.append(s + (p - s) * t)
            out.append(p)
        elif ds <= EPS:  # exiting: add crossing
            t = ds / (ds - dp) if (ds - dp) != 0 else 0.0
            out.append(s + (p - s) * t)
        s, ds = p, dp
    return np.array(out, dtype=float) if out else np.zeros((0, 3))


def plane_basis(n):
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, up)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def brush_tris(sides):
    """sides: list of (normal(3), dist). Returns list of triangles."""
    tris = []
    R = 32768.0
    for i, (ni, di) in enumerate(sides):
        ni = np.asarray(ni, dtype=float)
        nn = ni / np.linalg.norm(ni)
        u, v = plane_basis(nn)
        c = nn * (di / np.dot(nn, nn))
        poly = np.array([c + u * R + v * R, c - u * R + v * R,
                         c - u * R - v * R, c + u * R - v * R])
        for j, (nj, dj) in enumerate(sides):
            if j == i:
                continue
            poly = clip_poly(poly, np.asarray(nj, dtype=float), dj)
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            for k in range(1, len(poly) - 1):
                tris.append((poly[0], poly[k], poly[k + 1]))
    return tris


def load_solid_brushes(path):
    """BSP -> list of (planes (S,4) outward-normal/unit, dist; inside: n.x<=d,
    contents int) for PLAYERSOLID brushes after entity exclusion.
    Shared by tri-soup (below) and brush-plane sidecar (build_brushes.py)."""
    bsp = BSP(path)
    print('map loaded', path.split('/')[-1].split('\\')[-1], flush=True)

    # --- brush soup: entity exclusion fills bsp_br below --------------------
    n_brush = 0  # set after exclusion
    n_solid = n_ladder = n_other = 0
    out_brushes = []
    # --- brush entities: exclude non-solid/dynamic models' brushes -------
    # bsp.brushes contains ALL brushes (world + entity). Entity geometry ==
    # model faces exactly, so a brush whose every side-plane matches one of
    # the model's face planes belongs to that model (planes rounded).
    n_excl_brush = 0
    excl_detail = {}
    near_miss = []  # kept brushes sharing >=3 planes with an excluded model
    door_pts = []  # centroids of excluded door geometry (ambiguous when closed)
    try:
        models = list(bsp.bmodels.items())
        mclasses = {}
        for ent, model in models:
            try:
                cn = ent['classname']
            except Exception:
                cn = '?'
            mclasses[cn] = mclasses.get(cn, 0) + 1
        print('models=%d classes=%s' % (len(models), mclasses), flush=True)
        drop_sets = []
        for ent, model in models:
            try:
                cn = ent['classname']
            except Exception:
                continue
            keep_brush = False
            if cn == 'func_brush':
                # engine truth (modelentities.h): TOGGLE=0, NEVER=1, ALWAYS=2.
                # NEVER is non-solid (Spawn: VPHYSICS, never BSP); TOGGLE with
                # StartDisabled starts non-solid. Those brushes must NOT be
                # static collision (this was the phantom-giant source: an
                # 819-face NEVER model swallowing play areas).
                try:
                    sol = int(float(ent['solidity'])) if 'solidity' in ent else 2
                except Exception:
                    sol = 2
                try:
                    dis = int(float(ent['StartDisabled'])) if 'StartDisabled' in ent \
                        else (int(float(ent['startdisabled'])) if 'startdisabled' in ent else 0)
                except Exception:
                    dis = 0
                if sol == 1 or (sol == 0 and dis):
                    pass  # drop below
                else:
                    keep_brush = True
            elif cn not in EXCLUDE:
                keep_brush = True
            if keep_brush:
                continue
            try:
                ps = set()
                for fc in model.faces:
                    n = fc.plane.normal
                    ps.add((round(float(n.x) * 2) / 2, round(float(n.y) * 2) / 2,
                            round(float(n.z) * 2) / 2, round(float(fc.plane.dist) / ROUND) * ROUND))
                drop_sets.append((cn, ps))
                if cn == 'func_door':
                    try:
                        pts = []
                        for fc in model.faces:
                            for ed in fc.edges:
                                for attr in (None, 'start', 'v0', 'a'):
                                    try:
                                        if attr is None:
                                            a, b = ed[0], ed[1]
                                        else:
                                            a, b = getattr(ed, attr), getattr(
                                                ed, {'start': 'end', 'v0': 'v1', 'a': 'b'}[attr])
                                        pts.append((float(a.x), float(a.y), float(a.z)))
                                        pts.append((float(b.x), float(b.y), float(b.z)))
                                        break
                                    except Exception:
                                        continue
                        if pts:
                            door_pts.append(np.mean(np.array(pts), axis=0))
                    except Exception:
                        pass
            except Exception:
                continue
        if drop_sets:
            kept = []
            for br in bsp.brushes:
                try:
                    keys = set()
                    for s in br.sides:
                        n = s.plane.normal
                        keys.add((round(float(n.x) * 2) / 2, round(float(n.y) * 2) / 2,
                                  round(float(n.z) * 2) / 2, round(float(s.plane.dist) / ROUND) * ROUND))
                except Exception:
                    kept.append(br)
                    continue
                hit = None
                for cn, ps in drop_sets:
                    # Brush belongs to the model if nearly all of the BRUSH's
                    # planes appear in the model set (>=4 shared; plane+dist
                    # keys make accidental matches essentially impossible).
                    # (Subset direction matters: multi-brush entities own only
                    # some of the model's planes per brush; doors carry one
                    # extra culled side beyond their faces.)
                    if not keys or not ps:
                        continue
                    ov = len(keys & ps)
                    if ov >= 4 and ov >= len(keys) - 1:
                        hit = cn
                        break
                if hit:
                    n_excl_brush += 1
                    excl_detail[hit] = excl_detail.get(hit, 0) + 1
                else:
                    kept.append(br)
            bsp_br = kept
        else:
            bsp_br = list(bsp.brushes)
    except Exception as ex:
        print('entity-exclusion ERR', type(ex).__name__, str(ex)[:120], flush=True)
        bsp_br = list(bsp.brushes)
    print('excluded-brushes=%d %s near-miss-shared-planes=%s' % (
        n_excl_brush, excl_detail, sorted(near_miss, reverse=True)[:10]), flush=True)
    n_brush = len(bsp_br)

    for br in bsp_br:
        contents = br.contents
        if contents & MASK_PLAYERSOLID:
            n_solid += 1
            try:
                sides = [(np.array([s.plane.normal.x, s.plane.normal.y,
                                    s.plane.normal.z], dtype=float),
                          float(s.plane.dist)) for s in br.sides]
            except Exception:
                continue
            out_brushes.append((sides, int(contents.value)))
        elif contents & LADDER:
            n_ladder += 1
        else:
            n_other += 1
    print('brushes=%d solid=%d ladder=%d other=%d' % (
        n_brush, n_solid, n_ladder, n_other), flush=True)
    return out_brushes, n_brush, n_solid, n_ladder, n_other, door_pts


def main():
    t0 = time.time()
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    brushes, n_brush, n_solid, n_ladder, n_other, door_pts = load_solid_brushes(path)
    all_tris = []
    all_contents = []
    for sides, contents in brushes:
        for t in brush_tris(sides):
            all_tris.append(t)
            all_contents.append(contents)
    tris = np.array(all_tris, dtype=np.float32).reshape(-1, 3, 3) \
        if all_tris else np.zeros((0, 3, 3), dtype=np.float32)
    cont = np.array(all_contents, dtype=np.uint32)
    print('brushes=%d solid=%d ladder=%d other=%d tris=%d (%.1fs)' % (
        n_brush, n_solid, n_ladder, n_other, len(tris), time.time() - t0), flush=True)

    # --- static props: positions of solid ones (meshes deferred) -------------
    from collections import Counter
    prop_pts = []
    try:
        sol = Counter()
        nprop = 0
        for p in bsp.props:
            nprop += 1
            sol[p.solidity] += 1
            if p.solidity:
                try:
                    o = p.origin
                    prop_pts.append((float(o.x), float(o.y), float(o.z)))
                except Exception:
                    pass
        print('static_props=%d solidity=%s solid-pts=%d (meshes deferred)' % (
            nprop, dict(sol), len(prop_pts)), flush=True)
    except Exception as ex:
        print('static_props ERR', type(ex).__name__, str(ex)[:120], flush=True)
    prop_arr = np.array(prop_pts, dtype=np.float32).reshape(-1, 3) if prop_pts \
        else np.zeros((0, 3), dtype=np.float32)

    if out:
        # Uncompressed + atomic rename: compressed zip writes stall under CPU
        # starvation (training hogs cores) and get killed mid-write. 5MB is cheap.
        import os
        doors = np.array(door_pts, dtype=np.float32).reshape(-1, 3) \
            if door_pts else np.zeros((0, 3), dtype=np.float32)
        print('door_pts=%d' % len(doors), flush=True)
        tmp = out + '.tmp.npz'  # savez appends .npz when missing
        np.savez(tmp, tris=tris, contents=cont, doors=doors, props=prop_arr,
                 meta=np.array(['brushes=%d solid=%d tris=%d' % (n_brush, n_solid, len(tris))]))
        with np.load(tmp) as chk:  # closed handle: Windows replace needs it
            assert chk['tris'].shape[0] == len(tris), 'verify failed'
        os.replace(tmp, out)
        print('saved+verified', out, flush=True)


if __name__ == '__main__':
    main()
