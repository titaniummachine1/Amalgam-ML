"""Per-tick FORCE mask for behavior training (supervision mask, NOT row drop).

History always moves forward per-player, chronologically ordered; rows are
kept, warmup intact. The net is simply not rewarded/penalized on horizon
ticks where external forces act (knockback ballistics are out of scope).

F[i,h] = False (don't supervise) iff horizon tick T+h is inside the transient
of an external impulse: own victim damage [v, v+POST_F]. Airblast detector
exists but DEFAULT OFF (STV upsampling false-fires; MGE ~zero pyros).

Output: trainDforce.npz / evalDforce.npz with F bool (Nkeys, 66), aligned to
full Bk keys (same contract as Cmask/Dexcl).
"""
import csv
import os
import sqlite3
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict

import numpy as np

import didreg as _dr

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DB = os.path.join(DS, os.environ.get('ML_DB', 'dataset_all.db'))
THRESH = float(os.environ.get('AB_DV', '600.0'))
POST_F = int(os.environ.get('FORCE_POST', '8'))
AB_DETECT = os.environ.get('AB_DETECT', '0') == '1'
DT = 1 / 66.0

_did2base = _dr.reverse(_dr.load())
did2base = _did2base

print('resolving victim damage ticks...', flush=True)
vic = defaultdict(set)
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
                        vic[(base, ent)].add(t)
                        break
                else:
                    continue
                break
print(f'victim events: {sum(len(v) for v in vic.values())}', flush=True)
vic_arr = {k: np.array(sorted(v)) for k, v in vic.items()}

ab = defaultdict(set)
if AB_DETECT:
    print('detecting airblast impulses (diagnostic)...', flush=True)
    con = sqlite3.connect(DB)
    demos = [r[0] for r in con.execute('SELECT DISTINCT demo FROM ticks')]
    n_land = n_gap = 0
    for did in demos:
        rows = con.execute(
            'SELECT ent,tick,ox,oy,oz,grounded FROM ticks WHERE demo=? ORDER BY ent,tick',
            (did,)).fetchall()
        cur_ent = None
        tk, oo, gr = [], [], []
        def flush_series():
            global_n = None
            if len(tk) < 4:
                return
            # dedupe by tick (STV repeats positions)
            seen = set()
            utk, uo, ugr = [], [], []
            for t_, o_, g_ in zip(tk, oo, gr):
                if t_ in seen:
                    continue
                seen.add(t_)
                utk.append(t_)
                uo.append(o_)
                ugr.append(g_)
            if len(utk) < 4:
                return
            tk_a = np.array(utk)
            o_a = np.array(uo)
            gr_a = np.array(ugr)
            dt = np.diff(tk_a) * DT
            v = np.diff(o_a, axis=0) / dt[:, None]
            dv = np.diff(v, axis=0)
            mag = np.linalg.norm(dv, axis=1)
            base = did2base.get(did)
            hurt = vic.get((base, cur_ent[0]), set()) if base else set()
            for j in range(len(mag)):
                if mag[j] < THRESH or j < 2:
                    continue
                # sentinel: >300hu single-tick jump is garbage, not physics
                if (np.abs(o_a[j + 1] - o_a[j]).max() > 300.0
                        or np.abs(o_a[j + 2] - o_a[j + 1]).max() > 300.0):
                    continue
                if (tk_a[j + 1] - tk_a[j] != 1 or tk_a[j] - tk_a[j - 1] != 1):
                    continue
                gj = gr_a[j:j + 3]
                if len(gj) >= 2 and gj[0] < 0.5 and gj[1:].max() > 0.5:
                    continue
                if any(abs(tk_a[j + 1] - h) <= 4 for h in hurt):
                    continue
                ab[(did, cur_ent[0])].add(int(tk_a[j + 1]))
        for r in rows:
            if cur_ent is None or r[0] != cur_ent[0]:
                if cur_ent is not None:
                    flush_series()
                cur_ent = (r[0],)
                tk, oo, gr = [], [], []
            tk.append(r[1])
            oo.append((r[2], r[3], r[4]))
            gr.append(r[5])
        if cur_ent is not None:
            flush_series()
    print(f'airblast candidates: {sum(len(v) for v in ab.values())}', flush=True)
else:
    print('airblast detector OFF; force mask = damage transient only', flush=True)


def force_arr_for(did, ent):
    base = did2base.get(did)
    out = []
    if base is not None:
        va = vic_arr.get((base, ent))
        if va is not None and len(va):
            out.append(va)
    aa = ab.get((did, ent))
    if aa:
        out.append(np.array(sorted(aa)))
    if not out:
        return None
    return np.array(sorted(set(np.concatenate(out).tolist())))


def filt(tag):
    from collections import defaultdict as dd
    k = np.load(os.path.join(DS, tag + 'Bk.npz'))['K']
    K = np.array(k)
    F = np.ones((len(K), 66), dtype=bool)
    groups = dd(list)
    for i in range(len(K)):
        groups[(int(K[i][0]), int(K[i][1]))].append(i)
    for (did, ent), idxs in groups.items():
        fa = force_arr_for(did, ent)
        if fa is None:
            continue
        for i in idxs:
            t = int(K[i][2])
            lo = t + 1 - POST_F
            j0 = np.searchsorted(fa, lo)
            j1 = np.searchsorted(fa, t + 66, side='right')
            for f in fa[j0:j1]:
                h0 = max(f - t, 1)
                h1 = min(f + POST_F - t, 66)
                if h0 <= h1:
                    F[i, h0 - 1:h1] = False
    print(f'{tag}: masked horizon-ticks frac={(~F).mean():.4f} rows-touched={(~F.any(axis=1)).mean():.4f}',
          flush=True)
    np.savez_compressed(os.path.join(DS, tag + 'Dforce.npz'), F=F)


if __name__ == '__main__':
    filt('train')
    filt('eval')
