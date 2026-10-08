"""Context group (ctx): long-term memory without data explosion + opponents + heuristic.
Per sample (keyed demo/ent/T):
  EMA(15): trailing multi-timescale (tau=0.25/1/4s) means of accel(2D), vel(2D),
           speed(1) from origin-derived series. O(1) live update, fixed size.
  OPP(15): 3 nearest-enemy slots x (dir_ego(2), dist_clip(1), present(1)),
           rotated into sample ego frame. MGE: slots 2,3 empty (33p-ready).
  HEUR(2): yaw-rate heuristic accel (per-tick, ego frame, normalized).
Output C aligned 1:1 with train3/ev3 rows. Live cost: EMAs ~30flops/tick/player,
opp slots from entity loop already iterated by aimbot.
"""
import csv
import glob
import os
import numpy as np
import sqlite3
from collections import defaultdict

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DT = 1 / 66.0
TAUS = [0.25, 1.0, 4.0]
MAXV = 240.0


def load_opp(demo_base):
    """All alive players any class: tick -> list[(ent, x, y, team)]."""
    f = os.path.join(D, demo_base)
    by_tick = defaultdict(list)
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if row['life'] != '0':
                continue
            try:
                by_tick[int(row['tick'])].append(
                    (int(row['ent']), float(row['ox']), float(row['oy']), int(row['team'])))
            except ValueError:
                continue
    return by_tick


con = sqlite3.connect(os.path.join(DS, 'dataset.db'))

# did mapping identical to build_ds2: sorted csv list (excl. sidecars), index order
CSV_FILES = sorted(f for f in glob.glob(os.path.join(D, '*.csv')) if not f.endswith('.proj.csv'))
DID2CSV = {i: os.path.basename(f) for i, f in enumerate(CSV_FILES)}


def builder_yaw(ey_warm, vlt):
    if np.nanstd(ey_warm) < 0.5 and np.linalg.norm(vlt) > 50:
        return np.rad2deg(np.arctan2(vlt[1], vlt[0]))
    y = ey_warm[-1]
    return 0.0 if not np.isfinite(y) else y


def build(tag):
    k = np.load(os.path.join(DS, tag + '3k.npz'))
    K = k['K']
    N = len(K)
    E = np.zeros((N, 32), dtype=np.float32)
    # group sample indices by (did,ent); load each soldier series once (for EMA + frame)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(K[i][0]), int(K[i][1]))].append(i)
    # opponent tables per demo, loaded lazily
    opp_cache = {}

    def opp(did):
        if did not in opp_cache:
            opp_cache[did] = load_opp(DID2CSV[did])
        return opp_cache[did]

    for (did, ent), idxs in groups.items():
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
        # gap-aware derived velocity (hu/s): diff over actual tick gap; invalid across gaps
        gap = np.diff(tk, prepend=tk[0] - 1)
        vx = np.diff(ox, prepend=ox[0]) / np.maximum(gap, 1) / DT
        vy = np.diff(oy, prepend=oy[0]) / np.maximum(gap, 1) / DT
        ok_step = (gap == 1) & (np.hypot(np.diff(ox, prepend=ox[0]),
                                         np.diff(oy, prepend=oy[0])) <= 300.0)
        ok_step[0] = False
        # EMA states per tau: [ema_vx, ema_vy, ema_ax, ema_ay, ema_sp]; update on
        # consecutive valid steps only. Any gap (death / class change / teleport):
        # history invalid -> hard reset, never hold stale values.
        ema = {tau: np.zeros(5) for tau in TAUS}
        ema_map = {}
        pvx, pvy = 0.0, 0.0
        prev_ok = False
        for j in range(len(tk)):
            if not ok_step[j]:
                # death / class-change / teleport gap: history no longer valid -> hard reset
                for tau in TAUS:
                    ema[tau].fill(0.0)
                pvx, pvy = 0.0, 0.0
                prev_ok = False
                ema_map[int(tk[j])] = np.concatenate([ema[t] for t in TAUS]).copy()
                continue
            if ok_step[j]:
                dt = DT
                if prev_ok:
                    ax = (vx[j] - pvx) / dt
                    ay = (vy[j] - pvy) / dt
                else:
                    ax = ay = 0.0  # resume after gap: no stale-velocity spike
                sp = float(np.hypot(vx[j], vy[j]))
                for tau in TAUS:
                    al = 1.0 - np.exp(-dt / tau)
                    e = ema[tau]
                    e[0] += al * (vx[j] - e[0])
                    e[1] += al * (vy[j] - e[1])
                    e[2] += al * (ax - e[2])
                    e[3] += al * (ay - e[3])
                    e[4] += al * (sp - e[4])
                pvx, pvy = vx[j], vy[j]
                prev_ok = True
            ema_map[int(tk[j])] = np.concatenate([ema[t] for t in TAUS]).copy()
        # my team: from opponent CSV table (ticks table has no team column)
        ot = opp(did)
        myteam = -1
        for lst in ot.values():
            for (e2, _px, _py, tm) in lst:
                if e2 == ent:
                    myteam = tm
                    break
            if myteam >= 0:
                break
        for i in idxs:
            t = int(K[i][2])
            e15 = ema_map.get(t)
            if e15 is None:
                continue
            # normalize: vel/speed/accel by MAXV (accel per-sec norm, same as labels)
            E[i, 0:6] = e15[[0, 1, 5, 6, 10, 11]] / MAXV    # ema_v per tau
            E[i, 6:12] = np.clip(e15[[2, 3, 7, 8, 12, 13]] / MAXV, -15.0, 15.0)  # ema_a, knockback-clipped
            E[i, 12:15] = e15[[4, 9, 14]] / MAXV            # ema speed per tau
            # ego frame (builder rule) for opp dirs + heuristic
            jT = pos.get(t)
            if jT is None or jT < 66:
                continue
            wy = ey[jT - 65:jT + 1]
            vlt = np.array([vx[jT], vy[jT]])
            yaw = builder_yaw(wy, vlt)
            an = np.deg2rad(yaw)
            c, s = np.cos(-an), np.sin(-an)
            Re = np.array([[c, -s], [s, c]])
            # opponents: nearest 3 enemies at tick t (exact-tick snapshot)
            cands = [(np.hypot(px - ox[jT], py - oy[jT]), px, py)
                     for (e2, px, py, tm) in ot.get(t, []) if e2 != ent and tm != myteam]
            cands.sort()
            for sl in range(3):
                if sl < len(cands):
                    dd, px, py = cands[sl]
                    dv = np.array([px - ox[jT], py - oy[jT]])
                    n = max(np.linalg.norm(dv), 1.0)
                    E[i, 15 + sl * 3:15 + sl * 3 + 2] = Re @ (dv / n)
                    E[i, 15 + sl * 3 + 2] = min(dd / 2000.0, 1.0)
                    E[i, 24 + sl] = 1.0
            # heuristic: yaw-rate accel from last-8 derived headings (normalized /MAXV per-sec)
            j0 = max(jT - 7, 0)
            hh = np.arctan2(vy[j0:jT + 1], vx[j0:jT + 1])
            dh = (np.diff(hh) + np.pi) % (2 * np.pi) - np.pi
            w = dh.mean() if len(dh) else 0.0
            cw, sw = np.cos(w), np.sin(w)
            v0 = np.array([vx[jT], vy[jT]]) / MAXV
            ah = np.array([cw * v0[0] - sw * v0[1] - v0[0],
                           sw * v0[0] + cw * v0[1] - v0[1]]) * 66.0
            E[i, 27:29] = Re @ ah
            # slots present flags live at 24:27? layout: 0:15 ema, 15:24 opp(dir2+dist), 24:27 present, 27:29 heur, 29:32 spare
            E[i, 29:32] = 0.0
    np.savez_compressed(os.path.join(DS, tag + '_ctx.npz'), E=E)
    print(tag, E.shape, flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
