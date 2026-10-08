"""Per-tick event flags + health: fight state as GRU input features.

FLAGS (N,66,6) float32, per warmup tick s (ticks [T-65..T]):
  [:,:,0] fired_near fresh projectile within 500hu (any team) - shots around me.
  [:,:,1] fired_mine fresh projectile, same team, within 500hu - I (probably)
                     fired. (MGE teams are per-arena; same-team+near = mine.)
  [:,:,2] hurt_me    victim-resolved hurt at this tick - I got hit.
  [:,:,3] hurt_near  any resolved hurt with victim within 500hu - fight
                     intensity right next to me.
  [:,:,4] hp_self    own health / 200 (missing -> 0).
  [:,:,5] hp_thr     nearest hostile's health / 200 (identity resolved once at
                     T by position+team, series read over warmup; missing -> 0).
Event presence, not timing clocks: the GRU latches recency itself, and flags
are far cheaper to build (no 66-tick backward scans).
Why no reload flag: reload state is not in the dumps (needs the weapon sidecar;
queued). Why no "damage dealt": attacker UID has no UID->ent map; fired_mine
is its proxy.
Shardable (TSF_DIDS/TSF_SHARD); merge via build_tsfpar.py (same pattern).
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import didreg as _dr

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
W = 66
FIRE_R = 500.0
NEAR_R2 = 500.0 ** 2
SUF = os.environ.get('OUT_SUFFIX', 'B')
DID2CSV = _dr.reverse(_dr.load())


def load_demo(did):
    base = DID2CSV[did]
    health = defaultdict(dict)
    team = defaultdict(dict)
    with open(os.path.join(D, base)) as fh:
        for row in csv.DictReader(fh):
            if row['life'] != '0':
                continue
            try:
                e, t = int(row['ent']), int(row['tick'])
                health[e][t] = int(row['health'])
                team[e][t] = int(row['team'])
            except ValueError:
                continue
    hurts = []
    hf = os.path.join(D, base + '.hurt.csv')
    if os.path.exists(hf):
        with open(hf) as fh:
            for row in csv.DictReader(fh):
                if row.get('kind') != 'hurt':
                    continue
                try:
                    hurts.append((int(row['tick']), int(row['a']), int(row['b']),
                                  int(row['c']), int(row['d']), int(row['e'])))
                except ValueError:
                    continue
    fresh = defaultdict(list)
    pf = os.path.join(D, base + '.proj.csv')
    if os.path.exists(pf):
        with open(pf) as fh:
            for row in csv.DictReader(fh):
                if row.get('fresh') != '1':
                    continue
                try:
                    fresh[int(row['tick'])].append((float(row['ox']), float(row['oy']),
                                                     int(row['team'])))
                except (ValueError, KeyError):
                    continue
    return health, team, hurts, fresh


def build(tag):
    import sqlite3
    con = sqlite3.connect(os.path.join(DS, os.environ.get('ML_DB', 'dataset_all.db')))
    k = np.load(os.path.join(DS, tag + SUF + 'k.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    N = len(Ks)
    shard = os.environ.get('TSF_SHARD', '')
    only = os.environ.get('TSF_DIDS', '')
    only_set = set(int(x) for x in only.split(',') if x) or None
    F = np.zeros((N, W, 6), dtype=np.float32)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(Ks[i][0]), int(Ks[i][1]))].append(i)
    by_demo = defaultdict(list)
    for key, idxs in groups.items():
        by_demo[key[0]].append((key[1], idxs))
    if only_set is not None:
        by_demo = {d: v for d, v in by_demo.items() if d in only_set}
    for did, ent_groups in sorted(by_demo.items()):
        health, team, hurts, fresh = load_demo(did)
        # victim resolution (health-drop join), all hurts in demo
        vic = {}  # (tick, vent)
        vics = defaultdict(list)
        for (t, vu, au, dmg, wid, posthp) in hurts:
            best = None
            for ent, hs in health.items():
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
            if best is not None:
                vic[(t, best[1])] = True
                vics[best[1]].append(t)
        # victims by tick (once per demo)
        vbt = defaultdict(list)
        for (vt, ve) in vic.keys():
            vbt[vt].append(ve)
        # ONE batched position query per demo over the sampled span (per-sample
        # BETWEEN queries were 100k+ round trips). Dict-of-dicts, freed per demo.
        tmin = min(int(Ks[i][2]) for _, idxs in ent_groups for i in idxs) - 65
        tmax = max(int(Ks[i][2]) for _, idxs in ent_groups for i in idxs)
        Pall = defaultdict(dict)
        q = con.execute(
            'SELECT ent,tick,ox,oy FROM ticks WHERE demo=? AND tick BETWEEN ? AND ?',
            (did, tmin, tmax))
        for e2, tt, xx, yy in q.fetchall():
            Pall[e2][tt] = (xx, yy)
        for ent, idxs in ent_groups:
            rows = con.execute(
                'SELECT tick,ox,oy FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
                (did, ent)).fetchall()
            if not rows:
                continue
            tk = np.array([r[0] for r in rows])
            ox = np.array([r[1] for r in rows])
            oy = np.array([r[2] for r in rows])
            pos = {t: j for j, t in enumerate(tk)}
            my_tick_team = team.get(ent, {})
            # hostile identity at T: nearest non-self ent, hostile team preferred
            for i in idxs:
                t = int(Ks[i][2])
                jT = pos.get(t)
                if jT is None or jT < 65:
                    continue
                tm = my_tick_team.get(t, -1)
                # resolve threat ent at T from the demo-wide position cache
                P = Pall
                mine = P.get(ent, {})
                # threat identity at T
                thr, bd = -1, 1e18
                pt = mine.get(t)
                if pt is not None:
                    for e2, td in P.items():
                        if e2 == ent:
                            continue
                        q2 = td.get(t)
                        if q2 is None:
                            continue
                        if tm >= 0 and team.get(e2, {}).get(t, -2) == tm:
                            continue
                        dd = (q2[0] - pt[0]) ** 2 + (q2[1] - pt[1]) ** 2
                        if dd < bd:
                            bd, thr = dd, e2
                hh = health.get(ent, {})
                th = health.get(thr, {}) if thr >= 0 else {}
                for s in range(W):
                    tt = t - 65 + s
                    j = pos.get(tt)
                    if j is None:
                        continue
                    px, py = ox[j], oy[j]
                    # fresh projectiles at tt
                    fn = fired_mine = 0.0
                    for (fx, fy, ftm) in fresh.get(tt, []):
                        if (fx - px) ** 2 + (fy - py) ** 2 <= NEAR_R2:
                            fn = 1.0
                            if tm >= 0 and ftm == tm:
                                fired_mine = 1.0
                            elif tm < 0:
                                fired_mine = 1.0
                            break
                    hm = 1.0 if (tt, ent) in vic else 0.0
                    hn = 0.0
                    if hm == 0.0:
                        # any victim within 500hu at tt
                        for ve in vbt.get(tt, []):
                            qv = P.get(ve, {}).get(tt)
                            if qv is not None and (qv[0] - px) ** 2 + (qv[1] - py) ** 2 <= NEAR_R2:
                                hn = 1.0
                                break
                    else:
                        hn = 1.0
                    F[i, s, 0] = fn
                    F[i, s, 1] = fired_mine
                    F[i, s, 2] = hm
                    F[i, s, 3] = hn
                    F[i, s, 4] = hh.get(tt, 0) / 200.0
                    F[i, s, 5] = th.get(tt, 0) / 200.0
        print('  %s did %d' % (tag, did), flush=True)
    con.close()
    if shard:
        take = np.zeros(N, dtype=bool)
        for _ent, idxs in [p for v in by_demo.values() for p in v]:
            take[np.array(idxs, dtype=np.int64)] = True
        ii = np.where(take)[0]
        np.savez_compressed(shard, TSF=F[ii], IDX=ii)
        print('%s shard %s: %d rows' % (tag, shard, len(ii)), flush=True)
        return
    np.savez_compressed(os.path.join(DS, tag + SUF + '_tsf.npz'), TSF=F)
    print('%s TSF %s means: %s' % (tag, F.shape,
                                   np.round(F[:, 65, :].mean(axis=0), 4)), flush=True)


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'eval')
