"""Damage-tick EXCLUSION for behavior training (companion to build_combat.py).

Combat mask KEEPS damage neighborhoods (dodge behavior lives there), but the
damage ticks themselves must not be trained on: knockback is external force,
not input behavior - the target positions right after the impulse are
ballistics, and supervising the policy on them teaches it to hallucinate
explosions. (User directive 2026-10-08.)

Rule: label tick t is EXCLUDED iff the labeled player took damage (victim-
resolved, same resolver as build_combat) at tick v with t in [v-PRE, v+POST].
History (<=t) is observed truth and stays; the exclusion covers the impulse
transient corrupting the target horizon. Window env-tunable; defaults cover
the velocity flip (v..v+1) plus knockback decay.

Output: trainDexcl.npz / evalDexcl.npz with EXCL bool (True=exclude),
aligned to full Bk keys (same contract as Cmask). Separate file (not folded
into the mask) so exclusion can be ablated without rebuilding the mask.
Covers: explosions (damage-anchored). NOT covered: pyro airblast without a
damage event (no hurt row -> invisible; open item, needs cond-based detection).
"""
import csv
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import didreg as _dr
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
PRE = int(os.environ.get('DEX_PRE', '1'))    # exclude label ticks just before the impulse
POST = int(os.environ.get('DEX_POST', '6'))  # .. through the knockback transient

_did2base = _dr.reverse(_dr.load())  # did -> basename
did2base = _did2base

print('indexing damage events + resolving victims...', flush=True)
vic_ticks = defaultdict(set)   # (base, ent) -> set(tick)
n_res = n_fail = 0
for base in set(did2base.values()):
    health = defaultdict(dict)
    with open(os.path.join(D, base)) as fh:
        for row in csv.DictReader(fh):
            if row['life'] != '0':
                continue
            try:
                health[int(row['ent'])][int(row['tick'])] = int(row['health'])
            except ValueError:
                continue
    hf = os.path.join(D, base + '.hurt.csv')
    if not os.path.exists(hf):
        continue
    with open(hf) as fh:
        for row in csv.DictReader(fh):
            if row['kind'] != 'hurt':
                continue
            t, dmgv, posthp = int(row['tick']), int(row['c']), int(row['e'])
            for ent, hs in health.items():
                for dt in range(0, 4):
                    h = hs.get(t + dt)
                    if h is None:
                        continue
                    if h == posthp or (hs.get(t + dt - 1) is not None
                                       and hs[t + dt - 1] - h >= max(dmgv - 15, 1)):
                        vic_ticks[(base, ent)].add(t)
                        n_res += 1
                        break
                else:
                    continue
                break
            else:
                n_fail += 1
print(f'resolved={n_res} failed={n_fail}', flush=True)
vic_arr = {k: np.array(sorted(v)) for k, v in vic_ticks.items()}


def filt(tag):
    k = np.load(os.path.join(DS, tag + 'Bk.npz'))['K']
    K = np.array(k)
    excl = np.zeros(len(K), dtype=bool)
    for i in range(len(K)):
        base = did2base.get(int(K[i][0]))
        if base is None:
            continue
        t = int(K[i][2])
        ent = int(K[i][1])
        va = vic_arr.get((base, ent))
        if va is None or len(va) == 0:
            continue
        j = np.searchsorted(va, t)
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(va) and -PRE <= t - va[jj] <= POST:
                excl[i] = True
                break
    print(f'{tag}: excl {excl.sum()}/{len(K)} = {excl.mean():.4f}', flush=True)
    np.savez_compressed(os.path.join(DS, tag + 'Dexcl.npz'), EXCL=excl)


if __name__ == '__main__':
    filt('train')
    filt('eval')
