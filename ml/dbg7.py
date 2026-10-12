"""dbg7.py — miss forensics: full context + nearest player + multi-height rays."""
import csv
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from souptrace import load_soup, ray_hit

csvpath, souppath = sys.argv[1], sys.argv[2]
soup, _, doors = load_soup(souppath)
print('tris=%d doors=%d' % (len(soup), len(doors)), flush=True)
rows = defaultdict(list)
with open(csvpath, newline='') as fh:
    for r in csv.DictReader(fh):
        try:
            if int(float(r['class'])) in range(1, 10):
                rows[int(r['ent'])].append(r)
        except ValueError:
            continue
tickpos = defaultdict(dict)
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
shown = 0
for e, rs in rows.items():
    prev = None
    for i, r in enumerate(rs):
        t = int(r['tick'])
        p = np.array([float(r['ox']), float(r['oy']), float(r['oz'])])
        v = np.array([float(r['vx']), float(r['vy']), float(r['vz'])])
        gr = bool(int(float(r['flags'])) & 1)
        ms = float(r['mspd']) if r['mspd'] else 0.0
        if prev and t == prev['t'] + 1 and np.linalg.norm(p - prev['p']) <= 80 and ms > 1:
            s0 = float(np.hypot(prev['v'][0], prev['v'][1]))
            s1 = float(np.hypot(v[0], v[1]))
            if prev['gr'] and gr and s0 > 150 and s1 < 40:
                o = np.array([prev['p'][0], prev['p'][1], prev['p'][2] + 30.0])
                dd = np.array([prev['v'][0], prev['v'][1], 0.0])
                dd /= np.linalg.norm(dd)
                if ray_hit(o, dd, soup, s0 * 0.015 * 2 + 60.0) >= np.inf:
                    print('MISS ent=%d tick=%d p=(%.0f,%.0f,%.0f) v0=%.0f v1=%.0f ms=%.0f' % (
                        e, t, prev['p'][0], prev['p'][1], prev['p'][2], s0, s1, ms), flush=True)
                    for r2 in rs[max(0, i - 4):i + 2]:
                        print('  t=%s o=(%s,%s,%s) v=(%s,%s,%s) fl=%s hp=%s life=%s' % (
                            r2['tick'], r2['ox'], r2['oy'], r2['oz'], r2['vx'],
                            r2['vy'], r2['vz'], r2['flags'], r2.get('health'), r2.get('life')), flush=True)
                    others = [(f, q) for f, q in tickpos[t - 1].items() if f != e]
                    if others:
                        f, q = min(others, key=lambda fq: float(np.linalg.norm(fq[1] - prev['p'])))
                        print('  nearest player ent=%d d=%.0f at (%.0f,%.0f,%.0f)' % (
                            f, float(np.linalg.norm(q - prev['p'])), q[0], q[1], q[2]), flush=True)
                    for h in (0.0, 30.0, 60.0):
                        oh = np.array([prev['p'][0], prev['p'][1], prev['p'][2] + h])
                        print('  ray h=%.0f d500=%.0f' % (h, ray_hit(oh, dd, soup, 500.0)), flush=True)
                    shown += 1
                    if shown >= 3:
                        sys.exit(0)
        prev = {'t': t, 'p': p, 'v': v, 'gr': gr}
