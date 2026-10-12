"""diag_stops.py — why do wall-stop candidates miss the soup?
For each stop: tick (early/pre-round?), position cluster, nearest OTHER
player distance (body-block?). Verdict drives the fix (doors/pre-round skip,
player colliders, or props).
Usage: python ml/diag_stops.py <soup.npz> <csv> [max_checks]
"""
import csv
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from souptrace import load_soup, ray_hit

soup, _, _ = load_soup(sys.argv[1])
max_checks = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
rows = defaultdict(list)
with open(sys.argv[2], newline='') as fh:
    for r in csv.DictReader(fh):
        try:
            if int(float(r['class'])) in range(1, 10):
                rows[int(r['ent'])].append(r)
        except ValueError:
            continue
tickpos = defaultdict(dict)  # tick -> ent -> pos
for e in rows:
    rows[e].sort(key=lambda r: int(r['tick']))
    rs = rows[e]
    for a, b in zip(rs, rs[1:]):
        a['vx'] = (float(b['ox']) - float(a['ox'])) / 0.015
        a['vy'] = (float(b['oy']) - float(a['oy'])) / 0.015
        a['vz'] = (float(b['oz']) - float(a['oz'])) / 0.015
    rows[e] = rs[:-1]
    for r in rows[e]:
        tickpos[int(r['tick'])][e] = np.array(
            [float(r['ox']), float(r['oy']), float(r['oz'])])
stops = []
for e, rs in rows.items():
    prev = None
    for r in rs:
        t = int(r['tick'])
        p = np.array([float(r['ox']), float(r['oy']), float(r['oz'])])
        v = np.array([float(r['vx']), float(r['vy']), float(r['vz'])])
        gr = bool(int(float(r['flags'])) & 1)
        if prev and t == prev['t'] + 1 and np.linalg.norm(p - prev['p']) <= 300:
            s0 = float(np.hypot(prev['v'][0], prev['v'][1]))
            s1 = float(np.hypot(v[0], v[1]))
            if prev['gr'] and gr and s0 > 150 and s1 < 40:
                stops.append((t, e, prev['p'], prev['v'][:2] / (s0 + 1e-12), s0))
                if len(stops) >= max_checks:
                    break
        prev = {'t': t, 'p': p, 'v': v, 'gr': gr}
    if len(stops) >= max_checks:
        break
print('stops=%d' % len(stops), flush=True)
early = sum(1 for s in stops if s[0] < 3000)
print('early(t<3000)=%d (%.2f)' % (early, early / max(len(stops), 1)), flush=True)
nmiss = nmiss_early = nmiss_player = 0
misspos = []
for (t, e, p, d2, s0) in stops:
    o = np.array([p[0], p[1], p[2] + 30.0])
    dd = np.array([d2[0], d2[1], 0.0])
    hit = ray_hit(o, dd, soup, s0 * 0.015 * 2 + 60.0)
    if hit >= np.inf:
        nmiss += 1
        nmiss_early += t < 3000
        misspos.append(p)
        others = [q for f, q in tickpos[t].items() if f != e]
        if others:
            dmin = min(float(np.linalg.norm(q - p)) for q in others)
            if dmin < 70:
                nmiss_player += 1
print('miss=%d early-miss=%d player-near-miss(<70hu)=%d' % (nmiss, nmiss_early, nmiss_player), flush=True)
if misspos:
    m = np.array(misspos)
    print('miss centroid=(%.0f, %.0f, %.0f) spread=(%.0f, %.0f, %.0f)' % (
        tuple(m.mean(axis=0)) + tuple(m.std(axis=0))), flush=True)
    print('miss z range: %.0f..%.0f' % (m[:, 2].min(), m[:, 2].max()), flush=True)
