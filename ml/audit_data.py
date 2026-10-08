"""Trash-data audit: NaNs, dupes, class anomalies, proj coverage, tick sanity.
Reads CSVs + DB directly. Prints verdicts."""
import csv
import glob
import math
import os
import sqlite3
from collections import Counter

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')

print('=== 1. player CSV scan (alive rows only, like builders) ===')
cls_c = Counter()
nan_o = 0
nan_eye = 0
nan_v = 0
bad_life = Counter()
n_alive = 0
team_vals = Counter()
files = sorted(f for f in glob.glob(os.path.join(D, '*.csv')) if not f.endswith('.proj.csv'))
for f in files:
    with open(f) as fh:
        for row in csv.DictReader(fh):
            lf = row.get('life')
            if lf is None:
                bad_life['NO_LIFE_COL'] += 1
                continue
            if lf != '0':
                bad_life['dead_' + lf] += 1
                continue
            n_alive += 1
            try:
                ox = float(row['ox']); oy = float(row['oy'])
                cls_c[int(row['class'])] += 1
                team_vals[row.get('team')] += 1
                if math.isnan(ox) or math.isnan(oy):
                    nan_o += 1
                if row.get('eyey') is not None and math.isnan(float(row['eyey'])):
                    nan_eye += 1
                vx = float(row['vx'])
                if math.isnan(vx):
                    nan_v += 1
            except (ValueError, KeyError, TypeError):
                bad_life['PARSE_FAIL'] += 1
print('alive rows:', n_alive)
print('classes:', dict(sorted(cls_c.items())))
print('NaN ox/oy rows:', nan_o)
print('NaN eyey rows:', nan_eye)
print('NaN vx rows:', nan_v)
print('teams:', dict(team_vals))
print('non-alive buckets:', dict(bad_life))

print('=== 2. DB dupe check (PK should enforce uniqueness) ===')
con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))
n = con.execute('SELECT COUNT(*) FROM ticks').fetchone()[0]
nd = con.execute('SELECT COUNT(*) FROM (SELECT DISTINCT demo,ent,tick FROM ticks)').fetchone()[0]
print('rows:', n, 'distinct keys:', nd, 'dupes:', n - nd)

print('=== 3. tick sanity per (demo,ent): neg/zero ticks, huge gaps ===')
neg = 0
huge = 0
nseries = 0
for (did, ent) in con.execute('SELECT DISTINCT demo,ent FROM ticks').fetchall():
    nseries += 1
    tk = [r[0] for r in con.execute(
        'SELECT tick FROM ticks WHERE demo=? AND ent=? ORDER BY tick', (did, ent))]
    if tk[0] <= 0:
        neg += 1
    for a, b in zip(tk, tk[1:]):
        if b - a > 1000:
            huge += 1
            break
print('series:', nseries, 'starting tick<=0:', neg, 'with gap>1000:', huge)

print('=== 4. projectile sidecar coverage ===')
for f in files:
    base = os.path.basename(f)
    pf = f + '.proj.csv'
    if not os.path.exists(pf):
        print(f'{base[:23]}: NO PROJ FILE')
        continue
    n_all = n_fresh = 0
    teams = Counter()
    classes = Counter()
    z_nan = 0
    with open(pf) as fh:
        for row in csv.DictReader(fh):
            n_all += 1
            if row.get('fresh') == '1':
                n_fresh += 1
            teams[row.get('team')] += 1
            classes[row.get('class')] += 1
            try:
                if math.isnan(float(row['oz'])):
                    z_nan += 1
            except (ValueError, KeyError, TypeError):
                pass
    print(f'{base[:23]}: rows={n_all} fresh={n_fresh} ({n_fresh / max(n_all,1):.2f}) '
          f'teams={dict(teams)} classes={len(classes)} zNaN_frac={z_nan / max(n_all,1):.2f}')
