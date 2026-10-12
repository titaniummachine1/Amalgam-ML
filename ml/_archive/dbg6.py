import csv, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, 'C:/gitProjects/amalgam/ml')
from souptrace import load_soup, ray_hit

csvpath, souppath = sys.argv[1], sys.argv[2]
soup, _, _ = load_soup(souppath)
print('soup tris=%d' % len(soup), flush=True)
rows = defaultdict(list)
with open(csvpath, newline='') as fh:
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

# first stop event with context
for e, rs in rows.items():
    prev = None
    for i, r in enumerate(rs):
        t = int(r['tick'])
        p = np.array([float(r['ox']), float(r['oy']), float(r['oz'])])
        v = np.array([float(r['vx']), float(r['vy']), float(r['vz'])])
        gr = bool(int(float(r['flags'])) & 1)
        if prev and t == prev['t'] + 1 and np.linalg.norm(p - prev['p']) <= 300:
            s0 = float(np.hypot(prev['v'][0], prev['v'][1]))
            s1 = float(np.hypot(v[0], v[1]))
            if prev['gr'] and gr and s0 > 150 and s1 < 40:
                print('STOP ent=%d tick=%d' % (e, t), flush=True)
                for r2 in rs[max(0, i - 6):i + 3]:
                    print('  t=%s o=(%s,%s,%s) v=(%s,%s,%s) fl=%s ms=%s hp=%s' % (
                        r2['tick'], r2['ox'], r2['oy'], r2['oz'],
                        r2['vx'], r2['vy'], r2['vz'], r2['flags'], r2['mspd'],
                        r2.get('health')), flush=True)
                # ray from pre-stop pos along motion
                o = np.array([prev['p'][0], prev['p'][1], prev['p'][2] + 30.0])
                dd = np.array([prev['v'][0], prev['v'][1], 0.0])
                dd /= np.linalg.norm(dd)
                th = ray_hit(o, dd, soup, 500.0)
                print('  ray-hit-dist=%.1f (inf=none within 500hu)' % th, flush=True)
                # nearest soup tris to the stop point (brute force, one-time)
                d2 = ((soup - prev['p']) ** 2).sum(axis=(1, 2))
                for k in np.argsort(d2)[:5]:
                    c = soup[k].mean(axis=0)
                    print('  near-tri c=(%.0f,%.0f,%.0f) d=%.0f' % (
                        c[0], c[1], c[2], np.sqrt(d2[k] / 3)), flush=True)
                sys.exit(0)
        prev = {'t': t, 'p': p, 'v': v, 'gr': gr}
