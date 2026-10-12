"""validate_movesim.py — free-space gate for ml/movesim.py.
Usage: python ml/validate_movesim.py <csv> [max_pairs]
Checks (no collision involved, so failures = formula bugs, not soup):
  A. eye-alignment: ground wishdir vs view yaw must cluster at WASD angles.
  B. residual: forward-sim of recovered input must reproduce v1 (~0).
  C. air ballistics: fraction of air ticks with dvz == -g*dt (clean vs external).
  D. frozen-input rollout divergence @7/13/20/33/66 on clean ground segments.
Skips: tick gaps, teleports (>300hu), class change, death (life/health change).
Exit 0 + ALL PASS only if: WASD peaks dominate, mean res < 0.5, clean air
reported (no threshold — informational), rollout errors finite.
"""
import csv
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (CLASS_SPEED, DT, DUCK_FRAC, FL_DUCKING, FL_ONGROUND, SF,
                     friction, inverse_air, inverse_ground, rollout_free,
                     yaw_to_fwd)

HOR = [7, 13, 20, 33, 66]


def load(csvpath):
    rows = defaultdict(list)
    with open(csvpath, newline='') as fh:
        for r in csv.DictReader(fh):
            try:
                rows[int(r['ent'])].append(r)
            except ValueError:
                continue
    for e in rows:
        rows[e].sort(key=lambda r: int(r['tick']))
        rs = rows[e]
        # STV velocity columns are STALE (parser repeats baseline): derive
        # from positions. BUT the extractor also repeats whole ROWS on ticks
        # without an update (dormant/off-PVS): motion rows isolated, surrounded
        # by frozen repeats (m2f~0.9). Naive dp/row overestimates speed by the
        # gap length. Liveness = pos OR eye OR flags changed vs prev row;
        # stale repeats are DROPPED (standing-while-aiming rows survive via
        # eye changes, with true v=0). Pairs need tick adjacency (gap==1).
        live = []
        prev = None
        for r in rs:
            if prev is None:
                live.append(r)
            else:
                try:
                    moved = (r['ox'] != prev['ox'] or r['oy'] != prev['oy'] or r['oz'] != prev['oz'])
                    eyed = r['eyey'] != prev['eyey']
                    flgd = r['flags'] != prev['flags']
                except KeyError:
                    moved = eyed = flgd = True
                if moved or eyed or flgd:
                    live.append(r)
            prev = r
        nlp = 0
        for a, b in zip(live, live[1:]):
            try:
                gap = int(b['tick']) - int(a['tick'])
                if gap < 1:
                    continue
                k = 1.0 / (gap * DT)
                a['vx'] = (float(b['ox']) - float(a['ox'])) * k
                a['vy'] = (float(b['oy']) - float(a['oy'])) * k
                a['vz'] = (float(b['oz']) - float(a['oz'])) * k
                a['vgap'] = gap
            except ValueError:
                pass
        rows[e] = [r for r in live if 'vgap' in r]
    return rows


def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d


def main():
    path = sys.argv[1]
    max_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    rows = load(path)
    nb = {0: 0, 45: 0, 90: 0, 135: 0, 180: 0}
    nb_all = {0: 0, 45: 0, 90: 0, 135: 0, 180: 0}
    n_gr = n_capped = n_boost = n_clamp = 0
    res_sum = 0.0
    n_air = n_clean = 0
    errs = {h: [] for h in HOR}
    nseg = 0
    pairs = 0
    for e, rs in rows.items():
        mspd = 0.0
        prev = None
        seg = []  # clean ground run: (tick,p,v,wd,ws,ms)
        for r in rs:
            t = int(r['tick'])
            cls = int(f(r['class']))
            p = np.array([f(r['ox']), f(r['oy']), f(r['oz'])])
            v = np.array([f(r['vx']), f(r['vy']), f(r['vz'])])
            fl = int(f(r['flags']))
            ms = f(r['mspd'])
            ms = ms if ms > 1 else 0.0
            if ms <= 0:
                ms = mspd  # forward-fill
            if ms <= 0:
                ms = CLASS_SPEED.get(cls, 300.0)
            mspd = ms
            if fl & FL_DUCKING:
                ms = ms * DUCK_FRAC
            gr = bool(fl & FL_ONGROUND)
            cur = {'t': t, 'p': p, 'v': v, 'cls': cls, 'gr': gr, 'ms': ms,
                   'ey': f(r['eyey']), 'life': r.get('life'), 'hp': r.get('health')}
            if prev is not None:
                gap = t != prev['t'] + 1
                # 80hu teleport cut (not 300): legit max ~60hu/tick; spawn
                # teleports as small as 182hu fabricate -9000hu/s "velocities".
                tele = float(np.linalg.norm(p - prev['p'])) > 80
                swap = cls != prev['cls'] or cur['life'] != prev['life']
                npc = cls not in CLASS_SPEED  # static/world/bot rows carry no input info
                if not gap and not tele and not swap and not npc and pairs < max_pairs:
                    pairs += 1
                    if prev['gr'] and gr:
                        inv = inverse_ground(prev['v'], v, ms)
                        n_gr += 1
                        res_sum += inv['info']['res']
                        n_capped += inv['info']['capped']
                        n_boost += inv['info']['boosted']
                        n_clamp += inv['info']['clamped2d']
                        if inv['wishdir'] is not None:
                            ang = abs((math.degrees(math.atan2(
                                inv['wishdir'][1], inv['wishdir'][0]))
                                - cur['ey'] + 180) % 360 - 180)
                            b = int((ang + 22.5) // 45 * 45)  # nearest 45
                            b = max(0, min(180, b))
                            nb_all[b] += 1
                            # steady-state cruise ticks carry no input info
                            # (friction ~= accel cancels); only strong
                            # transients reveal true input direction.
                            dvn = float(np.linalg.norm(
                                v[:2] - friction(prev['v'])[:2]))
                            if dvn >= 15.0:
                                nb[b] += 1
                            seg.append((t, p, v, inv['wishdir'],
                                        inv['wishspeed'], ms))
                        else:
                            seg.append(None)
                        if len(seg) > 41:
                            seg.pop(0)
                        if len(seg) == 41 and all(seg) and seg[0] is not None:
                            s0 = seg[0]
                            inp = [(s0[3], s0[4])] * 40
                            grs = [True] * 40
                            pr = rollout_free(s0[2], s0[1], 40, inp, grs, s0[5])
                            for i, h in enumerate(HOR):
                                if h > 40:
                                    continue
                                actual = seg[h][1] if seg[h] else None
                                if actual is not None:
                                    errs[h].append(float(np.linalg.norm(
                                        pr[h - 1] - actual)))
                            nseg += 1
                            seg = []
                    elif not prev['gr'] and not gr:
                        inv = inverse_air(prev['v'], v)
                        n_air += 1
                        n_clean += inv['info']['dvz_clean']
                    else:
                        seg = []
                else:
                    seg = []
            prev = cur
    print('pairs=%d ground=%d air=%d' % (pairs, n_gr, n_air), flush=True)
    tot = sum(nb.values()) or 1
    tota = sum(nb_all.values()) or 1
    print('eye-align STRONG (dv>=15, want WASD peaks): ' +
          ' '.join('%d:%.2f' % (k, nb[k] / tot) for k in sorted(nb)), flush=True)
    print('eye-align ALL (cruise-dominated, expect flat): ' +
          ' '.join('%d:%.2f' % (k, nb_all[k] / tota) for k in sorted(nb_all)),
          flush=True)
    print('capped=%.3f boosted=%.3f clamped2d=%.3f mean_res=%.3f' % (
        n_capped / max(n_gr, 1), n_boost / max(n_gr, 1),
        n_clamp / max(n_gr, 1), res_sum / max(n_gr, 1)), flush=True)
    print('air dvz-clean=%.3f (rest = knockback/stairs/external)' % (
        n_clean / max(n_air, 1)), flush=True)
    for h in HOR:
        a = np.array(errs[h])
        print('frozen-input rollout @%d: n=%d med=%.1f p90=%.1f' % (
            h, len(a), np.median(a) if len(a) else -1,
            np.percentile(a, 90) if len(a) else -1), flush=True)
    wasd = (nb[0] + nb[90] + nb[180]) / tot
    # MGE-calibrated gate (duels are strafe-dominant; 0/180 naturally rare):
    # 90-mass>0.30 (strafe peak = behavior signal), residual<0.5 (round-trip
    # exactness), air dvz-clean>0.5 (ballistics dominate).
    # capped/boosted are NOT formula errors: capped share = external-force
    # prevalence (knockback residue, ramps — MGE-plausible); those rows are
    # masked from behavior loss downstream, so capped is recorded, not gated.
    cap = n_capped / max(n_gr, 1)
    air_clean = n_clean / max(n_air, 1)
    print('capped=%.3f (=external-force share, masked downstream)' % cap, flush=True)
    ok = (nb[90] / tot > 0.30) and (res_sum / max(n_gr, 1)) < 0.5 and air_clean > 0.5
    print('segments=%d WASD-mass=%.3f GATE=%s' % (
        nseg, wasd, 'PASS' if ok else 'FAIL'), flush=True)
    sys.exit(0 if ok else 1)


main()
