"""Combat-anchored sample selection (no data deleted).
Keep a window for training iff its label tick is combat-relevant for that player:
  - victim-precise: within [T-330, T+1320] (5s before, 20s after) of damage TAKEN
    by this player (victim resolved via health-drop correlation), OR
  - demo-fight: within same window of ANY damage event in the demo (covers
    attackers/dealers + everyone in an active fight; MGE arenas are small).
Timer extends naturally: overlapping windows union (a new damage event inside the
window stretches coverage). History-validity (consecutive/teleport/class/AFK ramp)
already enforced by the base builders; this is an ADDITIONAL mask.
Victim resolution: health == posthp or drop ~= dmg within [T, T+3].
Input: trainB/evB.npz + Bk keys. Output: trainC/evC.npz (+Ck keys).
"""
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
PRE = 330    # 5s before dealing/taking
POST = 1320  # 20s after (extends on new damage via union)

import didreg as _dr
_did2base = _dr.reverse(_dr.load())  # did -> basename
did2base = _did2base

print('indexing damage events + resolving victims...', flush=True)
vic_ticks = defaultdict(set)   # (base, ent) -> set(tick)
demo_ticks = defaultdict(set)  # base -> set(tick)
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
            demo_ticks[base].add(t)
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
print(f'resolved={n_res} failed={n_fail} demos_with_events={len(demo_ticks)}', flush=True)
# sorted arrays for searchsorted
vic_arr = {k: np.array(sorted(v)) for k, v in vic_ticks.items()}
demo_arr = {b: np.array(sorted(v)) for b, v in demo_ticks.items()}


def in_window(ticks_sorted, t):
    if len(ticks_sorted) == 0:
        return False
    i = np.searchsorted(ticks_sorted, t)
    for j in (i - 1, i):
        if 0 <= j < len(ticks_sorted) and -PRE <= t - ticks_sorted[j] <= POST:
            return True
    return False


def filt(tag):
    k = np.load(os.path.join(DS, tag + 'Bk.npz'))['K']
    K = np.array(k)
    keep = np.zeros(len(K), dtype=bool)
    for i in range(len(K)):
        base = did2base.get(int(K[i][0]))
        if base is None:
            continue
        t = int(K[i][2])
        ent = int(K[i][1])
        va = vic_arr.get((base, ent), np.empty(0, dtype=np.int64))
        da = demo_arr.get(base, np.empty(0, dtype=np.int64))
        if in_window(va, t) or in_window(da, t):
            keep[i] = True
    print(f'{tag}: keep {keep.sum()}/{len(K)} = {keep.mean():.3f}', flush=True)
    # mask + keys only: training applies mask to trainB arrays (no data duplication)
    np.savez_compressed(os.path.join(DS, tag + 'Ck.npz'), K=K[keep])
    np.savez_compressed(os.path.join(DS, tag + 'Cmask.npz'), keep=keep)


filt('train')
filt('eval')
