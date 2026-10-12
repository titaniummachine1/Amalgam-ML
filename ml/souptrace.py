"""souptrace.py — ray vs soup + wall-stop agreement gate (no game, no engine).
v1: brute-force vectorized Moller-Trumbore (fine for gate-scale; full-dataset
  rollouts move to Rust+BVH next). Hull sweeps come with the Rust stepper.
Gate: recorded wall-stops (grounded, speed>150 -> <40 in one tick) must have
  soup geometry along the motion ray. Misses are sampled for cause analysis
  (props = known gap, doors = excluded, players = not in soup).
Usage: python ml/souptrace.py <soup.npz> <csv> [max_checks]
"""
import csv
import sys
from collections import defaultdict

import numpy as np


def load_soup(path):
    d = np.load(path, allow_pickle=True)
    doors = d['doors'].astype(np.float64) if 'doors' in d else np.zeros((0, 3))
    return d['tris'].astype(np.float64), d['contents'], doors


def ray_hit(orig, direction, tris, maxdist):
    """Nearest hit distance or inf. Vectorized over all tris."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    p = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    tvec = orig - v0
    u = np.einsum('ij,ij->i', tvec, p) * inv
    q = np.cross(tvec, e1)
    v = np.dot(q, d) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t <= maxdist)
    if not np.any(hit):
        return np.inf
    return float(np.min(t[hit]))


def main():
    soup, _, doors = load_soup(sys.argv[1])
    max_checks = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    print('soup tris=%d' % len(soup), flush=True)
    from movesim import load_csv_live
    rows = load_csv_live(sys.argv[2], classes=set(range(1, 10)))
    tickpos = defaultdict(dict)  # tick -> ent -> pos (body-block forensics)
    for e2, rs2 in rows.items():
        for r2 in rs2:
            tickpos[int(r2['tick'])][e2] = np.array(
                [float(r2['ox']), float(r2['oy']), float(r2['oz'])])
    stops = []
    dup = tot = 0
    for e, rs in rows.items():
        prev = None
        pprev = None
        t0ent = int(rs[0]['tick']) if rs else 0
        last_snap = -10 ** 9
        last_life = None
        for r in rs:
            t = int(r['tick'])
            p = np.array([float(r['ox']), float(r['oy']), float(r['oz'])])
            v = np.array([float(r['vx']), float(r['vy']), float(r['vz'])])
            gr = bool(int(float(r['flags'])) & 1)
            ms = float(r['mspd']) if r['mspd'] else 0.0
            life = r.get('life')
            if prev and t == prev['t'] + 1:
                dp = float(np.linalg.norm(p - prev['p']))
                tot += 1
                dup += dp < 0.01
                if dp > 80 or life != last_life:
                    last_snap = t  # teleport/respawn/death: quarantine 66 ticks
            # Detector (all artifacts excluded):
            # - 80hu teleport cut (182hu spawn snaps fabricate -9000hu/s vels)
            # - 66-tick post-snap quarantine (respawn jitter, unset maxspeed)
            # - first 100 ticks per ent skipped (spawn transients)
            # - mspd>1 required (extractor must have seen live maxspeed)
            # - sustained pre-speed (snap-then-frozen artifacts)
            # - 150hu door exclusion (doors excluded from soup; may be closed)
            if prev and t == prev['t'] + 1 and np.linalg.norm(p - prev['p']) <= 80 \
                    and t - last_snap > 66 and t - t0ent > 100 and ms > 1 \
                    and r.get('vgap') == 1 and prev.get('vgap') == 1:
                s0 = float(np.hypot(prev['v'][0], prev['v'][1]))
                s1 = float(np.hypot(v[0], v[1]))
                spre = float(np.hypot(pprev['v'][0], pprev['v'][1])) if pprev and pprev['t'] == prev['t'] - 1 else 0.0
                if prev['gr'] and gr and s0 > 150 and s1 < 40 and spre > 100:
                    if len(doors) == 0 or float(np.min(((doors - prev['p']) ** 2).sum(axis=1)) ** 0.5) > 150:
                        stops.append((t, e, prev['p'], prev['v'][:2] / (s0 + 1e-12), s0))
                        if len(stops) >= max_checks:
                            break
            if prev and t == prev['t'] + 1:
                last_life = life
            pprev = prev
            prev = {'t': t, 'p': p, 'v': v, 'gr': gr, 'vgap': r.get('vgap')}
        if len(stops) >= max_checks:
            break
    print('wall-stop candidates=%d dup-row-frac=%.3f' % (len(stops), dup / max(tot, 1)), flush=True)
    # Post-filter: stop must PERSIST 2 ticks (kills single-row parser glitches
    # and dup artifacts) and the entity must be LIVE (eye moves within +-3
    # ticks; dormant freezes repeat eye verbatim). True wall-pins satisfy both.
    import bisect
    tickidx = {}
    for e2, rs2 in rows.items():
        tickidx[e2] = [int(r['tick']) for r in rs2]
    kept = []
    neye = 0
    for (st, se, p, d2, s0) in stops:
        rsx = rows[se]
        tx = tickidx[se]
        i = bisect.bisect_left(tx, st + 1)
        if i >= len(rsx) or tx[i] != st + 1:
            continue
        r1 = rsx[i]
        try:
            s_next = float(np.hypot(float(r1['vx']), float(r1['vy'])))
        except ValueError:
            continue
        if s_next >= 60:
            continue
        lo = bisect.bisect_left(tx, st - 3)
        eyes = []
        for r2 in rsx[lo:i + 4]:
            try:
                eyes.append(float(r2['eyey']))
            except ValueError:
                pass
        if not eyes or max(eyes) - min(eyes) <= 0.5:
            neye += 1
            continue
        kept.append((st, se, p, d2, s0))
    print('persistent-live stops=%d/%d (eye-dead-skipped=%d)' % (len(kept), len(stops), neye), flush=True)
    stops = kept
    if not stops:
        print('GATE=FAIL (no stops found)');
        sys.exit(1)
    hits = 0
    miss = []
    nnear = 0
    for i, (st, se, p, d2, s0) in enumerate(stops):
        o = np.array([p[0], p[1], p[2] + 30.0])  # chest height
        dd = np.array([d2[0], d2[1], 0.0])
        th = ray_hit(o, dd, soup, s0 * 0.015 * 2 + 60.0)
        if th < np.inf:
            hits += 1
        else:
            others = [q for f, q in tickpos.get(st, {}).items() if f != se]
            dmin = min([float(np.linalg.norm(q - p)) for q in others] or [1e9])
            if dmin < 100.0:
                nnear += 1  # body-block suspect: another player within 100hu
            elif len(miss) < 8:
                miss.append((p, d2))
        if (i + 1) % 500 == 0:
            print('  %d/%d hits=%d' % (i + 1, len(stops), hits), flush=True)
    agr = hits / len(stops)
    print('agreement=%.3f (%d/%d) body-near-miss=%d GATE=%s' % (
        agr, hits, len(stops), nnear, 'PASS' if agr > 0.90 else 'FAIL'), flush=True)
    for p, d in miss:
        print('  miss at (%.0f, %.0f, %.0f) dir (%.2f, %.2f)' % (p[0], p[1], p[2], d[0], d[1]), flush=True)
    sys.exit(0 if agr > 0.90 else 1)


if __name__ == '__main__':
    main()
