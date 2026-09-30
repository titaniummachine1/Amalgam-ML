"""Per-tick threat windows for training input (group: threatwin).
Same 6 features as build_ext.py per-tick definitions (ndist, nspd, infov, cnt,
tti, mem), but gathered over each sample's 66-tick warmup -> (N,66,6).
Aligned 1:1 with trainB/evalB SOLDIER rows (same order as dense cache).
DB: dataset_all.db (positions/eye), CSVs (teams), .proj.csv fresh rows (threats).
Output: trainB_threatwin.npz / evalB_threatwin.npz with TW (N,66,6).
"""
import csv
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import didreg
import sqlite3
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DT = 1 / 66.0
TELEPORT = 300.0
MEM_WIN = 66
PROJ_DT_TOL = 2
con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))

CSV_FILES = None  # registry-owned (see DID2CSV)
import didreg as _dr
DID2CSV = _dr.reverse(_dr.load())  # did -> basename, stable forever


def load_proj(demo_base):
    d = defaultdict(list)
    pf = os.path.join(D, demo_base + '.proj.csv')
    if not os.path.exists(pf):
        return d
    with open(pf) as fh:
        for row in csv.DictReader(fh):
            if row.get('fresh') != '1':
                continue
            try:
                d[int(row['tick'])].append((float(row['ox']), float(row['oy']),
                                            float(row['vx']), float(row['vy']),
                                            int(row['team'])))
            except (ValueError, KeyError):
                continue
    return d


def team_map(demo_base):
    votes = defaultdict(list)
    with open(os.path.join(D, demo_base)) as fh:
        for row in csv.DictReader(fh):
            try:
                votes[int(row['ent'])].append(int(row['team']))
            except (ValueError, KeyError):
                continue
    return {e: max(set(vs), key=vs.count) for e, vs in votes.items()}


def load_players(demo_base):
    """tick -> list[(ent, x, y, team)] alive rows, any class. For teammate slots."""
    d = defaultdict(list)
    with open(os.path.join(D, demo_base)) as fh:
        for row in csv.DictReader(fh):
            if row['life'] != '0':
                continue
            try:
                d[int(row['tick'])].append((int(row['ent']), float(row['ox']),
                                            float(row['oy']), int(row['team'])))
            except (ValueError, KeyError):
                continue
    return d


def build(tag):
    k = np.load(os.path.join(DS, tag + 'Bk.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    N = len(Ks)
    TW = np.zeros((N, 66, 6), dtype=np.float32)
    TS = np.zeros((N, 66, 5), dtype=np.float32)  # raw sim state: relpos ego(2), vel ego(2), present(1)
    TM = np.zeros((N, 66, 3, 4), dtype=np.float32)  # teammate slots: relpos world(2), dist(1), present(1)
    # defaults = no-info (match build_ext.py inst default)
    TW[:, :, 0] = 1.0
    TW[:, :, 5] = 1.0
    groups = defaultdict(list)
    for r in range(N):
        groups[(int(Ks[r][0]), int(Ks[r][1]))].append(r)
    by_demo = defaultdict(list)
    for key, idxs in groups.items():
        by_demo[key[0]].append((key[1], idxs))
    for did, ent_groups in sorted(by_demo.items()):
        demo_base = DID2CSV[did]
        proj = load_proj(demo_base)
        teams = team_map(demo_base)
        plocs = load_players(demo_base)
        for ent, idxs in ent_groups:
            rows = con.execute(
                'SELECT tick,ox,oy,eyey FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
                (did, ent)).fetchall()
            if not rows:
                continue
            tk = np.array([r[0] for r in rows])
            ox = np.array([r[1] for r in rows])
            oy = np.array([r[2] for r in rows])
            ey = np.array([r[3] for r in rows])
            pos = {t: j for j, t in enumerate(tk)}
            gap = np.diff(tk, prepend=tk[0] - 1)
            step = np.hypot(np.diff(ox, prepend=ox[0]), np.diff(oy, prepend=oy[0]))
            ok = (gap == 1) & (step <= TELEPORT)
            ok[0] = False
            vx = np.diff(ox, prepend=ox[0]) / np.maximum(gap, 1) / DT
            vy = np.diff(oy, prepend=oy[0]) / np.maximum(gap, 1) / DT
            sp = np.hypot(vx, vy)
            hd = np.arctan2(vy, vx)
            valid_hd = (sp > 50.0) & ok
            fwd_src = np.where(valid_hd, np.arange(len(tk)), -1)
            fwd = np.maximum.accumulate(fwd_src)
            fwd = np.clip(fwd, 0, len(tk) - 1)
            hd_ff = hd[fwd]
            hd_ok = fwd_src[np.clip(fwd, 0, len(tk) - 1)] >= 0
            myteam = teams.get(ent, -1)
            # needed ticks: union of warmup ranges over samples
            need = np.zeros(len(tk), dtype=bool)
            for r in idxs:
                t = int(Ks[r][2])
                jT = pos.get(t)
                if jT is None or jT < 66:
                    continue
                need[max(0, jT - 65):jT + 1] = True
            # per-tick instant threat + RAW sim state (world-frame relpos/vel of nearest
            # hostile; rotated into ego@T per sample below for rollout simulation)
            inst = {}
            raw = {}
            seen = np.zeros(len(tk), dtype=bool)
            for j in np.where(need)[0]:
                t = int(tk[j])
                best = None
                cnt = 0
                for dtb in range(0, PROJ_DT_TOL + 1):
                    lst = proj.get(t - dtb)
                    if not lst:
                        continue
                    for (px, py, pvx, pvy, ptm) in lst:
                        if ptm == myteam:
                            continue
                        dd = float(np.hypot(px - ox[j], py - oy[j]))
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
                raw[j] = (px - ox[j], py - oy[j],
                          pvx if np.isfinite(pvx) else 0.0,
                          pvy if np.isfinite(pvy) else 0.0)
                ndist = min(dd / 2000.0, 1.0)
                nspd = min(float(np.hypot(pvx, pvy)) / 3000.0, 1.0) \
                    if np.isfinite(pvx) and np.isfinite(pvy) else 0.0
                infov = 0.0
                if hd_ok[j]:
                    ang = np.arctan2(py - oy[j], px - ox[j])
                    dh = (ang - hd_ff[j] + np.pi) % (2 * np.pi) - np.pi
                    infov = 1.0 if abs(dh) < np.pi / 4 else 0.0
                tti = 1.0
                if np.isfinite(pvx) and np.isfinite(pvy) and dd > 1.0:
                    appr = -((pvx * (px - ox[j]) + pvy * (py - oy[j])) / dd)
                    if appr > 50.0:
                        tti = min(dd / appr / 5.0, 1.0)
                inst[j] = (ndist, nspd, infov, min(cnt, 8) / 8.0, tti)
                # seen-memory contributor: hostile <1500hu in cone
                if hd_ok[j]:
                    for dtb in range(0, PROJ_DT_TOL + 1):
                        for (qx, qy, _vx, _vy, ptm) in proj.get(t - dtb, []):
                            if ptm == myteam:
                                continue
                            if np.hypot(qx - ox[j], qy - oy[j]) >= 1500.0:
                                continue
                            ang = np.arctan2(qy - oy[j], qx - ox[j])
                            dh = (ang - hd_ff[j] + np.pi) % (2 * np.pi) - np.pi
                            if abs(dh) < np.pi / 4:
                                seen[j] = True
                                break
                        if seen[j]:
                            break
            cs = np.concatenate([[0], np.cumsum(seen.astype(np.int64))])
            idx_all = np.arange(len(tk))
            mem = (cs[np.clip(idx_all + 1, 0, len(tk))]
                   - cs[np.clip(idx_all - MEM_WIN, 0, len(tk))]) > 0
            for r in idxs:
                t = int(Ks[r][2])
                jT = pos.get(t)
                if jT is None or jT < 66:
                    continue
                if tk[jT] - tk[jT - 65] != 65:
                    continue  # warmup slice must be consecutive (matches base builder)
                # ego frame at T (same fallback rule as build_ds_all)
                vltT = np.array([ox[jT] - ox[jT - 1], oy[jT] - oy[jT - 1]]) / DT
                wyT = ey[jT - 65:jT + 1]
                yawT = ey[jT]
                if not np.isfinite(yawT):
                    yawT = 0.0
                if np.nanstd(wyT) < 0.5 and np.linalg.norm(vltT) > 50:
                    yawT = np.rad2deg(np.arctan2(vltT[1], vltT[0]))
                anT = np.deg2rad(yawT)
                cT, sT = np.cos(-anT), np.sin(-anT)
                ReT = np.array([[cT, -sT], [sT, cT]])
                for s in range(66):
                    jj = jT - 65 + s
                    if jj in inst:
                        ndist, nspd, infov, cnt, tti = inst[jj]
                        TW[r, s, 0] = ndist
                        TW[r, s, 1] = nspd
                        TW[r, s, 2] = infov
                        TW[r, s, 3] = cnt
                        TW[r, s, 5] = tti
                    TW[r, s, 4] = 1.0 if mem[jj] else 0.0
                    if jj in raw:
                        rx, ry, rvx, rvy = raw[jj]
                        TS[r, s, 0:2] = ReT @ np.array([rx, ry])
                        TS[r, s, 2:4] = ReT @ np.array([rvx, rvy])
                        TS[r, s, 4] = 1.0
                    # teammate slots: up to 3 same-team players within 1000hu
                    # (world-frame relpos + dist + present; rotate at train time like TS)
                    mates = []
                    for (e2, qx, qy, tm) in plocs.get(int(tk[jj]), []):
                        if e2 == ent or tm != myteam:
                            continue
                        dd = float(np.hypot(qx - ox[jj], qy - oy[jj]))
                        if dd <= 1000.0:
                            mates.append((dd, qx - ox[jj], qy - oy[jj]))
                    mates.sort()
                    for sl, (dd, dx, dy) in enumerate(mates[:3]):
                        TM[r, s, sl, 0] = dx
                        TM[r, s, sl, 1] = dy
                        TM[r, s, sl, 2] = dd / 1000.0
                        TM[r, s, sl, 3] = 1.0
    np.savez_compressed(os.path.join(DS, tag + 'B_threatwin.npz'), TW=TW, TS=TS, TM=TM)
    nz = (np.abs(TW).sum(axis=(1, 2)) > 0).mean()
    print(tag, TW.shape, 'frac rows with any signal: %.3f' % nz, flush=True)
    print(tag, 'col means:', TW.reshape(-1, 6).mean(axis=0).round(4), flush=True)
    print(tag, 'teammate present frac per slot:',
          (TM[:, :, :, 3] > 0).mean(axis=(0, 1)).round(3), flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
