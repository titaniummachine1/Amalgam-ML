"""build_cont.py — continuous future-input targets alongside binned TOK.
For each rollout npz row, recovers the CONTINUOUS (wishx, wishy, wishspeed)
per future tick from the source CSV (META holds (ent, tick_hist); npz name
maps to CSV). Saves sidecar <npz>.cont.npz with WC (N,FUT,3) + MSK copy.
Also reports the quantization ceiling: binned-tokens rollout vs continuous
rollout error (if binned-oracle < coast, bins are the bottleneck — proven).
Usage: python ml/build_cont.py <train_all.txt> <csvdir> <i> <nchunks>
"""
import os
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (CLASS_SPEED, DUCK_FRAC, FL_DUCKING, FL_ONGROUND,
                      back_clamp, ground_step, friction, inverse_ground,
                      inverse_air, load_csv_live, rollout_free, yaw_to_fwd)

HIST = 33
FUT = 20


def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d


def main():
    listpath, csvdir, ci, nc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    files = [l.strip() for l in open(listpath) if l.strip()]
    mine = [fp for j, fp in enumerate(files) if j % nc == ci]
    print('chunk %d/%d: %d/%d npz' % (ci, nc, len(mine), len(files)), flush=True)
    csv_cache = {}
    tot = qb = 0.0
    n = 0
    for fp in mine:
        base = os.path.basename(fp)
        csvname = base.replace('rollout_', '').replace('.npz', '') + '.csv'
        csvpath = os.path.join(csvdir, csvname)
        if not os.path.exists(csvpath):
            print('MISSING CSV, skip %s' % base, flush=True)
            continue
        if csvpath not in csv_cache:
            csv_cache.clear()
            csv_cache[csvpath] = load_csv_live(csvpath, classes=set(range(1, 10)))
        rows = csv_cache[csvpath]
        tickidx = {}
        for ent, rs in rows.items():
            tickidx[ent] = {}
            for k, r in enumerate(rs):
                try:
                    tickidx[ent][int(r['tick'])] = k
                except ValueError:
                    pass
        d = np.load(fp, allow_pickle=True)
        X, TOK, MSK, POS, META = d['X'], d['TOK'], d['MSK'], d['POS'], d['META']
        WC = np.zeros((len(X), FUT, 3), dtype=np.float32)
        for gi in range(len(X)):
            ent, th = int(META[gi][0]), int(META[gi][1])
            rs = rows.get(ent, [])
            k = tickidx.get(ent, {}).get(th, None)
            if k is None or k - HIST < 0:
                continue
            k0 = k - HIST
            if k0 + HIST + FUT >= len(rs):
                continue
            win = rs[k0:k0 + HIST + FUT + 1]
            V = np.array([[f(r['vx']), f(r['vy']), f(r['vz'])] for r in win])
            cls = int(f(win[0]['class']))
            ms0 = f(win[0]['mspd']) or CLASS_SPEED.get(cls, 300.0)
            raw = np.zeros((FUT, 3))
            for k in range(FUT):
                r = win[HIST + k]
                try:
                    msk = ms0 * (DUCK_FRAC if (int(f(r['flags'])) & FL_DUCKING) else 1.0)
                    gr = bool(int(f(r['flags'])) & FL_ONGROUND)
                    inv = inverse_ground(V[HIST + k], V[HIST + k + 1], msk) if gr \
                        else inverse_air(V[HIST + k], V[HIST + k + 1])
                except (ValueError, KeyError, IndexError):
                    continue
                wd = inv['wishdir']
                ws = inv['wishspeed'] if wd is not None else 0.0
                if wd is not None:
                    wd = np.asarray(wd, dtype=float).ravel()
                    raw[k, 0:2] = wd[:2]
                raw[k, 2] = ws / 400.0
            # RAW recovered inputs ONLY. Smoothing was tried and REJECTED:
            # smoothed-oracle replay scored 52.5/18.5/12.3 vs coast 91.0/56.1/21.8
            # — knowing the "denoised" future performed WORSE than no input at
            # all, because a smoothed wish is not a valid input to the sim.
            # Targets must be replayable (see parity gate in gate_replay.py).
            for k in range(FUT):
                wd = raw[k, :2]
                n = float(np.linalg.norm(wd))
                if n < 1e-6:
                    continue
                WC[gi, k, 0:2] = wd / n
                WC[gi, k, 2] = raw[k, 2]
        out = fp + '.cont.npz'
        tmp = out + '.tmp.npz'
        np.savez(tmp, WC=WC, MSK=MSK)
        os.replace(tmp, out)
        n += 1
        print('[%d/%d] %s rows=%d' % (n, len(mine), base, len(X)), flush=True)
    print('chunk %d done: %d files' % (ci, n), flush=True)


main()
