"""Knockback contamination measurement.
Join: hurt event (tick, victim_uid, dmg, post_hp) -> victim ent via health-drop
correlation in player CSVs. Then: what fraction of trainA soldier labels fall
within +-2 ticks of damage, and how do their |Y| compare to clean labels?
Also self vs enemy split, and damage->impulse calibration data (for cancel design)."""
import csv
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import didreg
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')

# 1. load hurt events per demo
hurts = defaultdict(list)   # demo_base -> list[(tick, vu, au, dmg, wid, posthp)]
deaths = defaultdict(list)
n_hurt = n_death = n_spawn = 0
for f in sorted(glob.glob(os.path.join(D, '*.csv'))):
    if f.endswith('.proj.csv') or f.endswith('.hurt.csv'):
        continue
    base = os.path.basename(f)
    hf = f + '.hurt.csv'
    if not os.path.exists(hf):
        continue
    with open(hf) as fh:
        for row in csv.DictReader(fh):
            if row['kind'] == 'hurt':
                hurts[base].append((int(row['tick']), int(row['a']), int(row['b']),
                                    int(row['c']), int(row['d']), int(row['e'])))
                n_hurt += 1
            elif row['kind'] == 'death':
                deaths[base].append((int(row['tick']), int(row['a']), int(row['b'])))
                n_death += 1
            else:
                n_spawn += 1
print(f'hurt={n_hurt} death={n_death} spawn={n_spawn}', flush=True)

# 2. victim resolution via health drops (per demo, alive rows)
res_ok = res_fail = 0
dmg_by_victim = defaultdict(list)  # (demo_base, ent) -> list[(tick, dmg, self)]
self_n = enemy_n = 0
for base, evs in hurts.items():
    # health series per ent
    health = defaultdict(dict)
    with open(os.path.join(D, base)) as fh:
        for row in csv.DictReader(fh):
            if row['life'] != '0':
                continue
            try:
                health[int(row['ent'])][int(row['tick'])] = int(row['health'])
            except ValueError:
                continue
    for (t, vu, au, dmg, wid, posthp) in evs:
        best = None
        for ent, hs in health.items():
            # look for health == posthp within [t, t+3], or a drop ~= dmg
            for dt in range(0, 4):
                h = hs.get(t + dt)
                if h is None:
                    continue
                if h == posthp or (hs.get(t + dt - 1) is not None
                                   and hs[t + dt - 1] - h >= max(dmg - 15, 1)):
                    score = abs(h - posthp)
                    if best is None or score < best[0]:
                        best = (score, ent)
                    break
        if best is None:
            res_fail += 1
        else:
            res_ok += 1
            _, ent = best
            dmg_by_victim[(base, ent)].append((t, dmg, vu == au))
            if vu == au:
                self_n += 1
            else:
                enemy_n += 1
print(f'resolved={res_ok} failed={res_fail} self={self_n} enemy={enemy_n}', flush=True)

# 3. flag trainA soldier labels within +-2 ticks of own damage
tr = np.load(os.path.join(DS, 'trainB.npz'), mmap_mode='r')
K = np.load(os.path.join(DS, 'trainBk.npz'))['K']
X, Y = np.array(tr['X']), np.array(tr['Y'])
# map demo did -> base name: did order == sorted csv list order (build_ds_all)
import didreg as _dr
did2base = _dr.reverse(_dr.load())  # did -> basename, stable forever
files = None  # registry-owned
# damage tick sets per (base, ent)
dmg_ticks = defaultdict(set)
for (base, ent), lst in dmg_by_victim.items():
    for (t, dmg, selfd) in lst:
        for dt in (-2, -1, 0, 1, 2):
            dmg_ticks[(base, ent)].add(t + dt)
flag = np.zeros(len(K), dtype=bool)
for i in range(len(K)):
    base = did2base.get(int(K[i][0]))
    if base is None:
        continue
    if int(K[i][2]) in dmg_ticks.get((base, int(K[i][1])), ()):
        flag[i] = True
ym = np.abs(Y).max(axis=1)
print(f'trainB rows={len(K)} flagged={flag.sum()} frac={flag.mean():.4f}', flush=True)
for name, m in [('clean', ~flag), ('flagged', flag)]:
    v = ym[m]
    print(f'{name}: n={m.sum()} p50={np.percentile(v,50):.2f} p90={np.percentile(v,90):.2f} '
          f'p99={np.percentile(v,99):.2f} max={v.max():.1f}', flush=True)
