"""build_rollout.py — counterfactual rollout dataset (transformer training rows).
Per window (66 history + 33 future live ticks, tick-adjacent):
  - history tokens (14 feats, world frame + eye + recovered inputs)
  - future: ego-frame input bins (THE TOKENIZER, 25 tokens) + jump flags +
    rel positions + rollout errors @8 horizons + per-tick loss-mask class
  - rollout_free with recovered ("sane demo") inputs -> predicted positions;
    innovation + lambda-blame computed in the TRAINER from stored errors.
Mask classes (loss weight): 0 clean (1.0), 1 capped lower-bound (0.5),
  2 flight/jump (0.0: WASD model predicts ground only; air stays context),
  3 blast/hp-drop +-5 (0.0). Teleports (dp>80) DROP the window (poison).
  Unpredictable ticks: never reward, never penalty — masked, context kept.
Tokenizer (ego frame, matches m_vAccel contract): none=0 (no input or
  ws<25); else octant(8) x speed-ternary(<150,>=150) -> 1..24; jump bit sep.
v1: open-space segments only (soup ray excludes wall-ahead; wall rows come
with traces later). Free-space sim; soup used for exclusion only.
History feats F=22: dxyz rel/1000, vxyz/1000, grounded, eye sin/cos,
mspd/400, class/9, rec wishdir world (2, 0 if none), rec ws/400,
ego hp/200, enemy rel xyz/1000 (3), enemy vel/1000 (3), enemy present (1).
Enemy = nearest same-tick other ent (opposite team preferred) — the duel
reactive signal ego kinematics cannot carry.
Usage: python ml/build_rollout.py <csv> <soup.npz> <out.npz> [stride]
"""
import math
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (AIRCAP, CLASS_SPEED, DUCK_FRAC, FL_DUCKING, FL_ONGROUND,
                     inverse_air, inverse_ground, load_csv_live, rollout_free,
                     yaw_to_fwd)
from souptrace import load_soup, ray_hit, build_grid, grid_hit

HIST = 33
FUT = 20
ERR_HOR = [4, 7, 10, 13, 17, 20]
FEATS = 22


def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d


def tokenize_full(wd, ws, eyedeg):
    """Ego-frame input bin (THE tokenizer — trainer + C++ port reuse it)."""
    if wd is None or ws < 25.0:
        return 0
    rel = (math.atan2(wd[1], wd[0]) - math.radians(eyedeg)) % (2 * math.pi)
    oct8 = int(rel // (math.pi / 4)) % 8
    spd = 1 if ws < 150.0 else 2
    return 1 + oct8 * 3 + (spd - 1)


def main():
    csvpath, souppath, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    rows = load_csv_live(csvpath, classes=set(range(1, 10)))
    soup, _, _ = load_soup(souppath)
    grid = build_grid(soup)
    print('grid cells=%d' % len(grid[2]), flush=True)
    # enemy join: per-ent tick->row maps (same-tick opponent state)
    tickmap = {}
    for oe, rs2 in rows.items():
        m = {}
        for r2 in rs2:
            try:
                m[int(r2['tick'])] = r2
            except ValueError:
                pass
        tickmap[oe] = m
    X, TOK, JMP, POS, ERR, MSK, META = [], [], [], [], [], [], []
    nwin = nwall = ntele = neye = 0
    mhist = [0, 0, 0, 0]
    thist = [0] * 25
    for e, rs in rows.items():
        i = 0
        L = HIST + FUT
        while i + L < len(rs):
            win = rs[i:i + L + 1]
            ticks = [int(r['tick']) for r in win]
            if any(b - a != 1 for a, b in zip(ticks, ticks[1:])):
                i += 1
                continue
            if any(int(f(r['class'])) != int(f(win[0]['class'])) for r in win):
                i += 1
                continue
            if any(f(r['mspd']) <= 1 for r in win):
                i += 1
                continue
            P = np.array([[f(r['ox']), f(r['oy']), f(r['oz'])] for r in win])
            if np.any(np.linalg.norm(P[1:] - P[:-1], axis=1) > 80):
                ntele += 1
                i += 1
                continue
            V = np.array([[f(r['vx']), f(r['vy']), f(r['vz'])] for r in win])
            cls = int(f(win[0]['class']))
            ms0 = f(win[0]['mspd']) or CLASS_SPEED.get(cls, 300.0)
            p0 = P[0]
            # eye-liveness gate (Catastrophe #2/#4 sequel): moving fast with
            # frozen eye = stale eye = poisoned ego bins. Slow windows keep
            # (eye irrelevant when still). Stale = <=2 unique eye values.
            try:
                eyen = len(set(r['eyey'] for r in win))
            except KeyError:
                eyen = 99
            if eyen <= 2:
                vbar = float(np.mean(np.linalg.norm(V[:HIST + 1, :2], axis=1)))
                if vbar > 100:
                    neye += 1
                    i += 1
                    continue
            # wall ahead? (open-only v1)
            sp0 = float(np.hypot(V[0][0], V[0][1]))
            if sp0 >= 50:
                dd = np.array([V[0][0] / sp0, V[0][1] / sp0, 0.0])
                o = np.array([P[0][0], P[0][1], P[0][2] + 30.0])
                plen = float(np.sum(np.linalg.norm(P[1:HIST + 1] - P[:HIST], axis=1)))
                if grid_hit(o, dd, soup, plen + 100.0, grid) < np.inf:
                    nwall += 1
                    i += HIST
                    continue
            # history features + recovered context inputs
            Hv = V[:HIST + 1]
            # raw per-tick history wishes (smoothed below: same INTENT DENOISE
            # as build_cont targets — net must read signal, not STV jitter)
            hwd = np.zeros((HIST, 2))
            hws = np.zeros(HIST)
            hfail = False
            for k in range(HIST):
                r = win[k]
                msk = ms0 * (DUCK_FRAC if (int(f(r['flags'])) & FL_DUCKING) else 1.0)
                gr = bool(int(f(r['flags'])) & FL_ONGROUND)
                try:
                    if gr:
                        inv = inverse_ground(Hv[k], Hv[k + 1], msk)
                    else:
                        inv = inverse_air(Hv[k], Hv[k + 1])
                except (ValueError, KeyError, IndexError):
                    hfail = True
                    break
                wd = inv['wishdir']
                if wd is not None:
                    hwd[k] = np.asarray(wd, dtype=float).ravel()[:2]
                    hws[k] = inv['wishspeed']
            if hfail:
                i += 1
                continue
            x = np.zeros((HIST, FEATS))
            ok = True
            egt = f(win[0].get('team'), -1.0)
            for k in range(HIST):
                r = win[k]
                try:
                    ey = f(r['eyey'])
                    msk = ms0 * (DUCK_FRAC if (int(f(r['flags'])) & FL_DUCKING) else 1.0)
                    gr = bool(int(f(r['flags'])) & FL_ONGROUND)
                    lo, hi = max(0, k - 2), min(HIST, k + 3)
                    mw = hwd[lo:hi].mean(axis=0)
                    nw = float(np.linalg.norm(mw))
                    ms = hws[lo:hi].mean()
                    x[k, 0:3] = (P[k] - p0) / 1000.0
                    x[k, 3:6] = Hv[k] / 1000.0
                    x[k, 6] = 1.0 if gr else 0.0
                    x[k, 7] = math.sin(math.radians(ey))
                    x[k, 8] = math.cos(math.radians(ey))
                    x[k, 9] = msk / 400.0
                    x[k, 10] = cls / 9.0
                    if nw >= 0.25:
                        x[k, 11:13] = mw / nw
                        x[k, 13] = ms / 400.0
                    x[k, 14] = f(r.get('health'), 200.0) / 200.0
                    # enemy reactive signal: nearest same-tick opponent
                    try:
                        tk = ticks[k]
                        best, bestd = None, 1e18
                        fallback, fbd = None, 1e18
                        for oe, m in tickmap.items():
                            if oe == e:
                                continue
                            orow = m.get(tk)
                            if orow is None:
                                continue
                            try:
                                ex = (f(orow['ox']) - P[k][0]) / 1000.0
                                eyy = (f(orow['oy']) - P[k][1]) / 1000.0
                                ezz = (f(orow['oz']) - P[k][2]) / 1000.0
                            except (ValueError, KeyError):
                                continue
                            dd = ex * ex + eyy * eyy + ezz * ezz
                            opp = (f(orow.get('team'), -2.0) != egt)
                            if opp and dd < bestd:
                                best, bestd = orow, dd
                            if dd < fbd:
                                fallback, fbd = orow, dd
                        erow = best if best is not None else fallback
                        if erow is not None:
                            x[k, 15] = (f(erow['ox']) - P[k][0]) / 1000.0
                            x[k, 16] = (f(erow['oy']) - P[k][1]) / 1000.0
                            x[k, 17] = (f(erow['oz']) - P[k][2]) / 1000.0
                            x[k, 18] = f(erow.get('vx'), 0.0) / 1000.0
                            x[k, 19] = f(erow.get('vy'), 0.0) / 1000.0
                            x[k, 20] = f(erow.get('vz'), 0.0) / 1000.0
                            x[k, 21] = 1.0
                    except (ValueError, KeyError, IndexError):
                        pass
                except (ValueError, KeyError, IndexError):
                    ok = False
                    break
            if not ok:
                i += 1
                continue
            # future: tokens + masks + rollout + errors
            Fv = V[HIST:HIST + FUT + 1]
            inp, views, toks, jmps, msks, grf = [], [], [], [], [], []
            for k in range(FUT):
                r = win[HIST + k]
                try:
                    ey = f(r['eyey'])
                except ValueError:
                    ey = 0.0
                views.append(yaw_to_fwd(ey))
                mskk = ms0 * (DUCK_FRAC if (int(f(r['flags'])) & FL_DUCKING) else 1.0)
                gr = bool(int(f(r['flags'])) & FL_ONGROUND)
                grf.append(gr)
                if gr:
                    inv = inverse_ground(Fv[k], Fv[k + 1], mskk)
                    mc = 1 if inv['info']['capped'] or inv['info']['boosted'] else 0
                else:
                    inv = inverse_air(Fv[k], Fv[k + 1])
                    mc = 2
                wd = inv['wishdir']
                ws = inv['wishspeed'] if wd is not None else 0.0
                inp.append((wd if wd is not None else None, ws))
                toks.append(tokenize_full(wd, ws, ey))
                try:
                    nxt_gr = bool(int(f(win[HIST + k + 1]['flags'])) & FL_ONGROUND)
                except (ValueError, IndexError):
                    nxt_gr = gr
                jmps.append(1 if gr and not nxt_gr else 0)
                msks.append(mc)
            # blast mask: hp DECREASE within +-5 ticks -> 0 loss (damage =
            # knockback = external force). Heals (increases) don't perturb
            # physics and stay clean.
            try:
                hps = [f(r.get('health')) for r in win]
                for k in range(FUT):
                    lo = max(0, HIST + k - 5)
                    hi = min(len(hps), HIST + k + 6)
                    w = hps[lo:hi]
                    if max(w) - min(w) > 0.5 and min(w) < hps[HIST + k] - 0.5:
                        msks[k] = 3
            except ValueError:
                pass
            pr = rollout_free(Fv[0], P[HIST], FUT, inp, grf, ms0, AIRCAP, views)
            errs = np.linalg.norm(pr - P[HIST + 1:HIST + FUT + 1], axis=1)
            X.append(x.astype(np.float32))
            TOK.append(np.array(toks, dtype=np.int64))
            JMP.append(np.array(jmps, dtype=np.int64))
            POS.append(((P[HIST + 1:HIST + FUT + 1] - P[HIST]) / 100.0).astype(np.float32))
            ERR.append(np.array([errs[h - 1] for h in ERR_HOR], dtype=np.float32))
            MSK.append(np.array(msks, dtype=np.int64))
            META.append(np.array([e, ticks[HIST]], dtype=np.int64))
            for m in msks:
                mhist[m] += 1
            for t in toks:
                thist[t] += 1
            nwin += 1
            i += stride
    print('windows=%d tele-dropped=%d wall-excluded=%d eye-dropped=%d' % (nwin, ntele, nwall, neye), flush=True)
    if nwin:
        Mt = sum(mhist)
        print('mask clean/capped/flight/blast: %s' % (
            ['%.2f' % (m / Mt) for m in mhist]), flush=True)
        Tt = sum(thist)
        print('tok none=%.2f moving=%d classes' % (thist[0] / Tt, sum(1 for t in thist[1:] if t)), flush=True)
        E = np.array(ERR)
        print('rollout-err med @%s: %s' % (ERR_HOR, np.median(E, axis=0).round(1).tolist()), flush=True)
        Xa, Ta, Ja, Pa, Ea, Ma, Me = (np.array(X), np.array(TOK), np.array(JMP),
                                      np.array(POS), E, np.array(MSK),
                                      np.array(META, dtype=np.int64))
    else:
        # dud demo (idle/empty): save EMPTY arrays with correct shapes so the
        # driver resume-check (len>0 required) keeps retrying rarely, and the
        # train list builder can filter by size. Exit 0 (not an error).
        Xa = np.zeros((0, HIST, FEATS), dtype=np.float32)
        Ta = np.zeros((0, FUT), dtype=np.int64)
        Ja = np.zeros((0, FUT), dtype=np.int64)
        Pa = np.zeros((0, FUT, 3), dtype=np.float32)
        Ea = np.zeros((0, len(ERR_HOR)), dtype=np.float32)
        Ma = np.zeros((0, FUT), dtype=np.int64)
        Me = np.zeros((0, 2), dtype=np.int64)
    np.savez(outpath if outpath.endswith('.npz') else outpath + '.npz',
             X=Xa, TOK=Ta, JMP=Ja, POS=Pa, ERR=Ea, MSK=Ma, META=Me)
    print('saved', outpath, flush=True)


main()
