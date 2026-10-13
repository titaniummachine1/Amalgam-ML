"""build_brushes.py — brush-plane sidecar for real swept-box traces (no game).
Loads solid PLAYERSOLID brushes (same exclusion as mapsoup) and saves each
brush's outward planes (unit normal + dist; inside: n.x <= d) + contents.
Output keys: bplanes (B,S,4) padded, bnsides (B,), bcontents (B,), bmins/bmaxs
(B,3) precomputed bounds, meta. Usage:
  python ml/build_brushes.py <bsp> <out_brushes.npz>
"""
import os
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from mapsoup import brush_tris, load_solid_brushes


def main():
    path, out = sys.argv[1], sys.argv[2]
    brushes, n_brush, n_solid, _, _, _ = load_solid_brushes(path)
    pls, cts = [], []
    for sides, contents in brushes:
        P = np.zeros((len(sides), 4), dtype=np.float64)
        for j, (n, d) in enumerate(sides):
            n = np.asarray(n, dtype=float)
            nl = float(np.linalg.norm(n))
            if nl < 1e-12:
                continue
            P[j, :3] = n / nl
            P[j, 3] = float(d) / nl
        # drop degenerate planes
        P = P[np.linalg.norm(P[:, :3], axis=1) > 1e-9]
        if len(P) == 0:
            continue
        pls.append(P)
        cts.append(contents)
    S = max(len(P) for P in pls) if pls else 0
    B = len(pls)
    # consistency filter (parser artifacts): keep a brush only if its planes
    # admit a strictly-inside corner (batched triple solve). Inconsistent sets
    # (empty intersection) can spuriously claim start-inside; the engine never
    # contains such brushes (vbsp rejects them). e.g. beta4b brush 6440: giant
    # claimed-solid box swallowing the play area, tri soup disagrees.
    from itertools import combinations as _comb
    _trips_cache = {}
    keep = []
    for i, P in enumerate(pls):
        n = len(P)
        ok = False
        # bevel monsters (S>24) skip the test: 59 agreeing planes cannot
        # falsely claim inside; the failure mode needs small inconsistent sets.
        if n > 24:
            ok = True
        elif n >= 4:
            if n not in _trips_cache:
                _trips_cache[n] = np.array(list(_comb(range(n), 3)), dtype=np.int64)
            trips = _trips_cache[n]
            M = P[trips, :3]
            det = np.linalg.det(M)
            good = np.abs(det) > 1e-9
            if np.any(good):
                # corner lies ON its 3 defining planes: exclude them from the
                # inside test (require all OTHER planes nd < +0.5).
                X = np.linalg.solve(M[good], P[trips[good], 3][..., None]).squeeze(-1)
                tg = trips[good]
                nd = X @ P[:, :3].T - P[:, 3][None, :]
                for gi in range(len(tg)):
                    row = nd[gi].copy()
                    row[tg[gi]] = -np.inf  # defining planes: on-boundary, ignore
                    if np.all(row < 0.5):
                        ok = True
                        break
        if ok:
            keep.append(i)
    print('consistency: kept %d / %d' % (len(keep), B), flush=True)
    pls = [pls[i] for i in keep]
    cts = [cts[i] for i in keep]
    B = len(pls)
    bp = np.zeros((B, S, 4), dtype=np.float32)
    bn = np.zeros((B,), dtype=np.int64)
    for i, P in enumerate(pls):
        bp[i, :len(P)] = P.astype(np.float32)
        bn[i] = len(P)
    bc = np.array(cts, dtype=np.uint32)
    # bounds per brush: min/max over the brush's own surface tris (exact —
    # tris ARE the brush surface; same builder as the tri soup, C-speed).
    bmin = np.full((B, 3), np.inf)
    bmax = np.full((B, 3), -np.inf)
    ndeg = 0
    for i in range(B):
        P = bp[i, :bn[i]].astype(np.float64)
        sides = [(P[j, :3], P[j, 3]) for j in range(len(P))]
        tl = [np.asarray(t, dtype=float).reshape(-1, 3)
              for t in brush_tris(sides)] if sides else []
        if tl:
            pts = np.concatenate(tl, axis=0)
            bmin[i] = pts.min(axis=0) - 0.5
            bmax[i] = pts.max(axis=0) + 0.5
        else:
            bmin[i] = -1e9  # degenerate: never cull, planes decide
            bmax[i] = 1e9
            ndeg += 1
    tmp = out + '.tmp.npz'
    np.savez(tmp, bplanes=bp, bnsides=bn, bcontents=bc, bmins=bmin.astype(np.float32),
             bmaxs=bmax.astype(np.float32),
             meta=np.array(['brushes=%d solid=%d' % (n_brush, n_solid)]))
    with np.load(tmp) as chk:
        assert chk['bplanes'].shape[0] == B, 'verify failed'
    os.replace(tmp, out)
    print('brushes=%d maxsides=%d degenerate=%d saved+verified %s' % (B, S, ndeg, out), flush=True)


main()
