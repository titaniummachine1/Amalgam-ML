"""Known-forces sidecar: per-sample external impulse vectors from hurt events.

Why: knockback is a KNOWN external force, not behavior. Instead of dropping
knockback samples (impulse-split) we keep them: the rollout applies the known
impulse as physics FIRST, the policy predicts input accel on top (at the bitter
end). Samples REMAIN trash (dropped by the cap backstop) only when the force is
UNKNOWN.

Output (aligned 1:1 with soldier rows of trainBk/evalBk, B or C via OUT_SUFFIX):
  KB  (N,66,4) future forces [kx, ky, dmg, has] in ego@T frame. kb[h] feeds
               rollout step h+1 (transition T+h -> T+h+1), i.e. the hurt at
               tick T+h. Zero where no hurt resolved. Matches the runtime that
               re-predicts every tick with fresh damage info.
  KBW (N,66,3) warmup forces [kx, ky, dmg] for ticks [T-65, T] (knockback
               history fills the GRU input slots).
Direction sources, in order (all in ego@T frame via the dataset frame rule):
  1. nearest HOSTILE projectile within 200hu at the hurt tick (explosives =
     the impulses that matter; team from the main CSV);
  2. nearest other player within 500hu (airblast/melee/hitscan - MGE arenas are
     >>500hu apart so this is the attacker; no team lookup needed; pure DB).
     There is NO networked airblast-victim flag (verified: m_hStunner is a
     constant invalid handle, stun flags always zero) - this fallback is what
     catches airblasts (big impulse, no damage event needed... note: airblast
     with zero damage produces no hurt event at all and stays unknown-force).
  3. self-damage (vu==au, rocket jumps): dir=0, dmg kept (model learns "blast
     happened"; horizontal RJ impulse is small/unknowable, vertical ignored).
Also prints K_fit: |observed dv - friction| regressed on dmg for directed
grounded hurts. Rollout uses impulse = dir * dmg * K_fit.

Victim resolution: health-drop join (health == posthp or drop >= max(dmg-15,1)
within [t, t+3], best match) - same rule as measure_knockback (~90% resolve).
"""
import csv
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import didreg as _dr

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
W = 66
SUF = os.environ.get('OUT_SUFFIX', 'B')
EYE_FRAME = os.environ.get('EYE_FRAME', '0') == '1'
PROJ_R = 200.0
MATE_R = 500.0
DB = os.path.join(DS, os.environ.get('ML_DB', 'dataset_all.db'))
DID2CSV = _dr.reverse(_dr.load())
con = sqlite3.connect(DB)


def ego_yaw(vxT, vyT, ey_hist, eyT):
    yaw = eyT if np.isfinite(eyT) else 0.0
    sp = float(np.hypot(vxT, vyT))
    if EYE_FRAME:
        if not np.isfinite(eyT):
            yaw = np.rad2deg(np.arctan2(vyT, vxT)) if sp > 1.0 else 0.0
    elif sp > 50.0:
        yaw = np.rad2deg(np.arctan2(vyT, vxT))
    return yaw


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
    proj = defaultdict(list)
    pf = os.path.join(D, base + '.proj.csv')
    if os.path.exists(pf):
        with open(pf) as fh:
            for row in csv.DictReader(fh):
                if row.get('fresh') != '1':
                    continue
                try:
                    proj[int(row['tick'])].append((float(row['ox']), float(row['oy']),
                                                   int(row['team'])))
                except (ValueError, KeyError):
                    continue
    # all player positions NEAR HURTS ONLY (for the nearest-mate fallback): the
    # full-demo tick scan this replaced cost ~3GB RAM per demo. Hurts are rare
    # (~hundreds/demo), so fetch ticks in [hurt-1, hurt+1] exclusively.
    mates = defaultdict(dict)
    if hurts:
        hticks = sorted(set(t + d for (t, *_r) in hurts for d in (-1, 0, 1)))
        q = con.execute(
            'SELECT ent,tick,ox,oy FROM ticks WHERE demo=? AND tick IN (%s)' % (
                ','.join('?' * len(hticks))), (did, *hticks))
        for e, t, x, y in q.fetchall():
            mates[e][t] = (x, y)
    return health, team, hurts, proj, mates


def resolve(health, hurts):
    out = []
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
            out.append((t, best[1], au, dmg, wid, vu == au))
    return out


def direct(vx, vy, R):
    d = R @ np.array([vx, vy])
    n = float(np.hypot(*d))
    if n < 1e-6:
        return 0.0, 0.0, False
    return d[0] / n, d[1] / n, True


def build(tag):
    k = np.load(os.path.join(DS, tag + SUF + 'k.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    N = len(Ks)
    # SHARD MODE (parallel builds): KNOCK_DIDS="3,7,11" processes only those
    # demos and saves a partial shard; merge_shards() assembles the full cache.
    # Row indices are stable (keys npz is final), so shards slot in exactly.
    shard = os.environ.get('KNOCK_SHARD', '')
    only = os.environ.get('KNOCK_DIDS', '')
    only_set = set(int(x) for x in only.split(',') if x) or None
    KB = np.zeros((N, W, 4), dtype=np.float32)
    KBW = np.zeros((N, W, 3), dtype=np.float32)
    idx_all = np.arange(N)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(Ks[i][0]), int(Ks[i][1]))].append(i)
    by_demo = defaultdict(list)
    for key, idxs in groups.items():
        by_demo[key[0]].append((key[1], idxs))
    if only_set is not None:
        by_demo = {d: v for d, v in by_demo.items() if d in only_set}
        if not by_demo:
            print('%s: shard has no demos, writing empty' % tag, flush=True)
            np.savez_compressed(shard or os.path.join(DS, tag + SUF + '_knock.npz'),
                                KB=np.zeros((0, W, 4), dtype=np.float32),
                                KBW=np.zeros((0, W, 3), dtype=np.float32),
                                IDX=np.zeros((0,), dtype=np.int64))
            return
    kfits = []
    n_hurt = n_res = n_dir = n_mate = 0
    h = np.arange(W)
    for did, ent_groups in sorted(by_demo.items()):
        health, team, hurts, proj, mates = load_demo(did)
        res = resolve(health, hurts)
        n_hurt += len(hurts)
        n_res += len(res)
        by_vic = defaultdict(list)
        for (t, vent, au, dmg, wid, selfd) in res:
            by_vic[vent].append((t, au, dmg, wid, selfd))
        for ent, idxs in ent_groups:
            rows = con.execute(
                'SELECT tick,ox,oy,eyey FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
                (did, ent)).fetchall()
            if not rows:
                continue
            tk = np.array([r[0] for r in rows])
            ox = np.array([r[1] for r in rows])
            oy = np.array([r[2] for r in rows])
            ey = np.array([r[3] for r in rows], dtype=np.float64)
            pos = {t: j for j, t in enumerate(tk)}
            ev = by_vic.get(ent, [])
            if not ev:
                continue
            for i in idxs:
                t = int(Ks[i][2])
                jT = pos.get(t)
                if jT is None or jT < 66:
                    continue
                vlt = np.array([ox[jT] - ox[jT - 1], oy[jT] - oy[jT - 1]]) * 66.0
                yaw = ego_yaw(vlt[0], vlt[1], ey[max(0, jT - 65):jT + 1], ey[jT])
                an = np.deg2rad(yaw)
                c, s = np.cos(-an), np.sin(-an)
                R = np.array([[c, -s], [s, c]])
                for (ht, au, dmg, wid, selfd) in ev:
                    jh = pos.get(ht)
                    if jh is None:
                        continue
                    dx, dy, known = 0.0, 0.0, False
                    if not selfd:
                        tm = team.get(ent, {}).get(ht, team.get(ent, {}).get(t, -1))
                        bd, bx, by = 1e18, 0.0, 0.0
                        for dtb in range(3):
                            for (px, py, ptm) in proj.get(ht - dtb, []):
                                if ptm == tm:
                                    continue
                                dd = float(np.hypot(px - ox[jh], py - oy[jh]))
                                if dd < bd:
                                    bd, bx, by = dd, px, py
                            if bd < 1e18:
                                break
                        if bd <= PROJ_R:
                            dx, dy, known = direct(ox[jh] - bx, oy[jh] - by, R)
                            if known:
                                n_dir += 1
                        if not known:
                            md, mx, my = 1e18, 0.0, 0.0
                            for e2, td in mates.items():
                                if e2 == ent:
                                    continue
                                q = td.get(ht)
                                if q is None:
                                    continue
                                dd = float(np.hypot(q[0] - ox[jh], q[1] - oy[jh]))
                                if dd < md:
                                    md, mx, my = dd, q[0], q[1]
                            if md <= MATE_R:
                                dx, dy, known = direct(ox[jh] - mx, oy[jh] - my, R)
                                if known:
                                    n_mate += 1
                    if t <= ht < t + W:
                        KB[i, ht - t, 0] = dx
                        KB[i, ht - t, 1] = dy
                        KB[i, ht - t, 2] = dmg
                        KB[i, ht - t, 3] = 1.0
                    if t - 65 <= ht <= t:
                        KBW[i, ht - (t - 65), 0] = dx
                        KBW[i, ht - (t - 65), 1] = dy
                        KBW[i, ht - (t - 65), 2] = dmg
                    if jh > 0 and jh + 1 < len(tk) and tk[jh + 1] == ht + 1:
                        dvx = (ox[jh + 1] - ox[jh]) * 66.0 - (ox[jh] - ox[jh - 1]) * 66.0
                        dvy = (oy[jh + 1] - oy[jh]) * 66.0 - (oy[jh] - oy[jh - 1]) * 66.0
                        kfits.append((float(np.hypot(dvx, dvy)), float(dmg),
                                      bool(known)))
        print('  %s did %d' % (tag, did), flush=True)
    if shard:
        take = np.zeros(N, dtype=bool)
        for _ent, idxs in [p for v in by_demo.values() for p in v]:
            take[np.array(idxs, dtype=np.int64)] = True
        ii = np.where(take)[0]
        np.savez_compressed(shard, KB=KB[ii], KBW=KBW[ii], IDX=ii)
        print('%s shard %s: %d rows' % (tag, shard, len(ii)), flush=True)
        return
    np.savez_compressed(os.path.join(DS, tag + SUF + '_knock.npz'), KB=KB, KBW=KBW)
    has = KB[:, :, 3].sum(axis=1) > 0
    print('%s N=%d rows w/ future hurt %.4f  w/ directed %.4f' % (
        tag, N, has.mean(),
        float((np.abs(KB[:, :, 0:2]).sum(axis=(1, 2)) > 0).mean())), flush=True)
    if kfits:
        a = np.array([(dv, dg) for (dv, dg, kn) in kfits if kn and dg > 0])
        if len(a) > 100:
            K = float((a[:, 0] * a[:, 1]).sum() / (a[:, 1] ** 2).sum())
            print('%s K_fit = %.3f hu/s per dmg (directed hurts n=%d)' % (tag, K, len(a)),
                  flush=True)
    print('%s hurts=%d resolved=%d proj-dir=%d mate-dir=%d' % (
        tag, n_hurt, n_res, n_dir, n_mate), flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'eval')
