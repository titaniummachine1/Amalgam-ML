"""Extended per-sample features (training mirror of runtime validity + threats).
Aligned 1:1 with trainA/evalA rows via Ak keys (did,ent,tick,cls).

W2 walk_age (N,): ticks since walk began looking back through consecutive
  same-class ticks (cap 264, /264 normalize; -1.0 sentinel = never moved).
  Mirrors C++ m_iSinceWalk / IsHistoryValid.

T threat (N,6), all 0..1: [nearest hostile dist, nearest hostile speed,
  nearest in-FOV90, count<1000hu, seen-in-FOV-last-66, min time-to-impact].
  Hostile = proj.team != player team (team map from player CSVs, majority vote).
  Projectiles from .proj.csv fresh==1 rows only (stale positions mislead);
  tick match within [t-2, t]. Eye yaw is stale in demos -> velocity-heading
  fallback when speed>50 (same rule as ego frame); frozen ticks use last
  valid heading (forward-filled). 2D only (rocket oz is NaN in dumps).

Threat computed per (demo,ent) series but ONLY for ticks near samples
(union of [label-66, label] ranges) to avoid 9.6M-row Python loops.
DB: dataset_all.db. Output trainA_ext.npz {W2, T} / evalA_ext.npz.
"""
import csv
import glob
import os
import sqlite3
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DT = 1 / 66.0
TELEPORT = 300.0
WALK_CAP = 264
MEM_WIN = 66
PROJ_DT_TOL = 2  # accept projectile snapshot within [t-2, t]

con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))

CSV_FILES = sorted(f for f in glob.glob(os.path.join(D, '*.csv')) if not f.endswith('.proj.csv'))
DID2CSV = {i: os.path.basename(f) for i, f in enumerate(CSV_FILES)}


def load_proj(demo_base):
    """tick -> list[(x,y,vx,vy,team)] fresh rows only."""
    d = defaultdict(list)
    with open(os.path.join(D, demo_base + '.proj.csv')) as fh:
        for row in csv.DictReader(fh):
            if row.get('fresh') != '1':
                continue
            try:
                t = int(row['tick'])
                d[t].append((float(row['ox']), float(row['oy']),
                             float(row['vx']), float(row['vy']), int(row['team'])))
            except (ValueError, KeyError):
                continue
    return d


def team_map(demo_base):
    """ent -> majority team from player CSV."""
    votes = defaultdict(list)
    with open(os.path.join(D, demo_base)) as fh:
        for row in csv.DictReader(fh):
            try:
                votes[int(row['ent'])].append(int(row['team']))
            except (ValueError, KeyError):
                continue
    out = {}
    for e, vs in votes.items():
        out[e] = max(set(vs), key=vs.count)
    return out


def build(tag):
    k = np.load(os.path.join(DS, tag + 'Ak.npz'))
    K = k['K']
    N = len(K)
    W2 = np.full(N, -1.0, dtype=np.float32)
    T = np.zeros((N, 6), dtype=np.float32)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(K[i][0]), int(K[i][1]))].append(i)
    # group series by demo for shared proj/team loads
    by_demo = defaultdict(list)
    for key, idxs in groups.items():
        by_demo[key[0]].append((key[1], idxs))
    for did, ent_groups in sorted(by_demo.items()):
        demo_base = DID2CSV[did]
        proj = load_proj(demo_base)
        teams = team_map(demo_base)
        for ent, idxs in ent_groups:
            rows = con.execute(
                'SELECT tick,ox,oy,eyey,cls FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
                (did, ent)).fetchall()
            if not rows:
                continue
            tk = np.array([r[0] for r in rows])
            ox = np.array([r[1] for r in rows])
            oy = np.array([r[2] for r in rows])
            ey = np.array([r[3] for r in rows])
            cl = np.array([r[4] for r in rows])
            pos = {t: j for j, t in enumerate(tk)}
            gap = np.diff(tk, prepend=tk[0] - 1)
            step = np.hypot(np.diff(ox, prepend=ox[0]), np.diff(oy, prepend=oy[0]))
            ok = (gap == 1) & (step <= TELEPORT) & (np.ediff1d(cl, to_begin=0) == 0)
            ok[0] = False
            vx = np.diff(ox, prepend=ox[0]) / np.maximum(gap, 1) / DT
            vy = np.diff(oy, prepend=oy[0]) / np.maximum(gap, 1) / DT
            sp = np.hypot(vx, vy)
            # heading with forward-fill for frozen ticks (vectorized)
            hd = np.arctan2(vy, vx)
            valid_hd = (sp > 50.0) & ok
            fwd_src = np.where(valid_hd, np.arange(len(tk)), -1)
            fwd = np.maximum.accumulate(fwd_src)
            fwd = np.clip(fwd, 0, len(tk) - 1)
            hd_ff = hd[fwd]
            hd_ok = fwd_src[np.clip(fwd, 0, len(tk) - 1)] >= 0  # has any past valid heading
            myteam = teams.get(ent, -1)
            # needed ticks: union of [label-66, label] over samples
            need = np.zeros(len(tk), dtype=bool)
            for i in idxs:
                jT = pos.get(int(K[i][2]))
                if jT is not None:
                    need[max(0, jT - MEM_WIN):jT + 1] = True
            # per-tick instant threat (loop only over needed ticks)
            inst = {}  # j -> (ndist, nspd, infov, cnt, tti)
            for j in np.where(need)[0]:
                t = int(tk[j])
                best = None
                cnt = 0
                tti_best = 1.0
                for dtb in range(0, PROJ_DT_TOL + 1):
                    lst = proj.get(t - dtb)
                    if not lst:
                        continue
                    for (px, py, pvx, pvy, ptm) in lst:
                        if ptm == myteam:
                            continue
                        dx, dy = px - ox[j], py - oy[j]
                        dd = float(np.hypot(dx, dy))
                        if dd < 1000.0:
                            cnt += 1
                        if best is None or dd < best[0]:
                            best = (dd, px, py, pvx, pvy)
                    if best is not None:
                        break
                if best is None:
                    inst[j] = (1.0, 0.0, 0.0, 0.0, 1.0)
                    continue
                dd, px, py, pvx, pvy = best
                ndist = min(dd / 2000.0, 1.0)
                nspd = min(float(np.hypot(pvx, pvy)) / 3000.0, 1.0) if np.isfinite(pvx) and np.isfinite(pvy) else 0.0
                infov = 0.0
                if hd_ok[j]:
                    ang = np.arctan2(py - oy[j], px - ox[j])
                    dh = (ang - hd_ff[j] + np.pi) % (2 * np.pi) - np.pi
                    infov = 1.0 if abs(dh) < np.pi / 4 else 0.0
                tti = 1.0
                if np.isfinite(pvx) and np.isfinite(pvy) and dd > 1.0:
                    # proj moving toward player <=> vel . (player - proj) > 0
                    appr = -((pvx * (px - ox[j]) + pvy * (py - oy[j])) / dd)
                    if appr > 50.0:
                        tti = min(dd / appr / 5.0, 1.0)
                inst[j] = (ndist, nspd, infov, min(cnt, 8) / 8.0, tti)
            # seen-memory: hostile within 1500hu AND in-cone within trailing 66 (vectorized flags)
            near = np.zeros(len(tk), dtype=bool)
            seen = np.zeros(len(tk), dtype=bool)
            for j in np.where(need)[0]:
                t = int(tk[j])
                hit = False
                for dtb in range(0, PROJ_DT_TOL + 1):
                    for (px, py, _vx, _vy, ptm) in proj.get(t - dtb, []):
                        if ptm == myteam:
                            continue
                        if np.hypot(px - ox[j], py - oy[j]) < 1500.0:
                            hit = True
                            break
                    if hit:
                        break
                near[j] = hit
                seen[j] = hit and bool(hd_ok[j]) and False  # cone needs per-proj angle; approx below
            # proper seen-memory with cone: recompute per needed tick against proj list
            for j in np.where(need)[0]:
                if not hd_ok[j]:
                    continue
                t = int(tk[j])
                found = False
                for dtb in range(0, PROJ_DT_TOL + 1):
                    for (px, py, _vx, _vy, ptm) in proj.get(t - dtb, []):
                        if ptm == myteam:
                            continue
                        if np.hypot(px - ox[j], py - oy[j]) >= 1500.0:
                            continue
                        ang = np.arctan2(py - oy[j], px - ox[j])
                        dh = (ang - hd_ff[j] + np.pi) % (2 * np.pi) - np.pi
                        if abs(dh) < np.pi / 4:
                            found = True
                            break
                    if found:
                        break
                seen[j] = found
            cs = np.concatenate([[0], np.cumsum(seen.astype(np.int64))])
            idx_all = np.arange(len(tk))
            # trailing-66 OR via cumsum on seen
            b = np.clip(idx_all + 1, 0, len(tk))
            mem = (cs[b] - cs[np.clip(idx_all - MEM_WIN, 0, len(tk))]) > 0
            # walk age: distance back to last moving tick within consecutive run
            moving = (step > 0.5) & ok
            # runs of ok: label walk-back bounded by run starts; vectorize per run
            for i in idxs:
                t = int(K[i][2])
                jT = pos.get(t)
                if jT is None:
                    continue
                # walk back while consecutive+ok to find first movement
                js = jT
                last_move = -1
                while js >= 0 and ok[js] and (jT - js) <= WALK_CAP:
                    if step[js] > 0.5:
                        last_move = js
                    js -= 1
                if last_move >= 0:
                    W2[i] = min(jT - last_move, WALK_CAP) / WALK_CAP
                else:
                    W2[i] = -1.0
                if jT in inst:
                    ndist, nspd, infov, cnt, tti = inst[jT]
                    T[i, 0] = ndist
                    T[i, 1] = nspd
                    T[i, 2] = infov
                    T[i, 3] = cnt
                    T[i, 4] = 1.0 if mem[jT] else 0.0
                    T[i, 5] = tti
    np.savez_compressed(os.path.join(DS, tag + '_ext.npz'),
                        W2=W2, T=T)
    print(tag, 'W2 valid frac: %.3f' % ((W2 >= 0).mean()),
          'threat nonzero frac: %.3f' % ((np.abs(T).sum(axis=1) > 0).mean()), flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
