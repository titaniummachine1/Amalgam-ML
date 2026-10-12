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
    rows = defaultdict(list)
    with open(sys.argv[2], newline='') as fh:
        for r in csv.DictReader(fh):
            try:
                if int(float(r['class'])) in range(1, 10):
                    rows[int(r['ent'])].append(r)
            except ValueError:
                continue
    for e in rows:
        rows[e].sort(key=lambda r: int(r['tick']))
        rs = rows[e]
        for a, b in zip(rs, rs[1:]):
            a['vx'] = (float(b['ox']) - float(a['ox'])) / 0.015
            a['vy'] = (float(b['oy']) - float(a['oy'])) / 0.015
            a['vz'] = (float(b['oz']) - float(a['oz'])) / 0.015
        rows[e] = rs[:-1]
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
                    and t - last_snap > 66 and t - t0ent > 100 and ms > 1:
                s0 = float(np.hypot(prev['v'][0], prev['v'][1]))
                s1 = float(np.hypot(v[0], v[1]))
                spre = float(np.hypot(pprev['v'][0], pprev['v'][1])) if pprev and pprev['t'] == prev['t'] - 1 else 0.0
                if prev['gr'] and gr and s0 > 150 and s1 < 40 and spre > 100:
                    if len(doors) == 0 or float(np.min(((doors - prev['p']) ** 2).sum(axis=1)) ** 0.5) > 150:
                        stops.append((prev['p'], prev['v'][:2] / (s0 + 1e-12), s0))
                        if len(stops) >= max_checks:
                            break
            if prev and t == prev['t'] + 1:
                last_life = life
            pprev = prev
            prev = {'t': t, 'p': p, 'v': v, 'gr': gr}
        if len(stops) >= max_checks:
            break
    print('wall-stop candidates=%d dup-row-frac=%.3f' % (len(stops), dup / max(tot, 1)), flush=True)
    if not stops:
        print('GATE=FAIL (no stops found)');
        sys.exit(1)
    hits = 0
    miss = []
    for i, (p, d2, s0) in enumerate(stops):
        o = np.array([p[0], p[1], p[2] + 30.0])  # chest height
        dd = np.array([d2[0], d2[1], 0.0])
        t = ray_hit(o, dd, soup, s0 * 0.015 * 2 + 60.0)
        if t < np.inf:
            hits += 1
        elif len(miss) < 8:
            miss.append((p, d2))
        if (i + 1) % 500 == 0:
            print('  %d/%d hits=%d' % (i + 1, len(stops), hits), flush=True)
    agr = hits / len(stops)
    print('agreement=%.3f (%d/%d) GATE=%s' % (
        agr, hits, len(stops), 'PASS' if agr > 0.90 else 'FAIL'), flush=True)
    for p, d in miss:
        print('  miss at (%.0f, %.0f, %.0f) dir (%.2f, %.2f)' % (p[0], p[1], p[2], d[0], d[1]), flush=True)
    sys.exit(0 if agr > 0.90 else 1)


if __name__ == '__main__':
    main()
