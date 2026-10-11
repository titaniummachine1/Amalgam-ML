"""Jump-flight mask for behavior-only training (B line, soldier rows).
Jump = takeoff (gr[t-1]==1 & gr[t]==0, tick-consecutive) with vz[t] > 150 hu/s
(census 2026-10-11: foot-jump spike 250-320, walked-off-ledge 0-50 kept,
stairs 50-150 kept, blast takeoffs 500+ also excluded as external impulse).
Flight = [takeoff, landing) (landing tick grounded -> supervised again).
JM (N,66) = 0 for horizon ticks inside a flight, 1 else. Fully-masked rows
contribute exactly 0 (mask-sum clamp). Eval untouched (measure everything).
Output trainB_jumpmask.npz {JM (N,66) bool}, soldier-ordered 1:1 with repack.
Usage: python ml/build_jumpmask.py"""
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
con = sqlite3.connect(os.path.join(D, 'dataset_all.db'))
DT = 1 / 66.0
VZ_JUMP = 150.0
W = 66


def build(tag):
    k = np.load(os.path.join(D, tag + 'Bk.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    N = len(Ks)
    JM = np.ones((N, W), dtype=bool)
    JH132 = np.ones((N, 132), dtype=bool)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(Ks[i][0]), int(Ks[i][1]))].append(i)
    done = n_rows_masked = n_ticks_masked = 0
    for (did, ent), idxs in groups.items():
        rows = con.execute(
            'SELECT tick,oz,grounded FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
            (did, ent)).fetchall()
        if len(rows) < 10:
            continue
        tk = np.array([r[0] for r in rows])
        oz = np.array([r[1] for r in rows])
        gr = np.array([r[2] for r in rows])
        pos = {t: j for j, t in enumerate(tk)}
        v = np.diff(oz, prepend=oz[0]) / DT
        # takeoffs: consecutive ticks, leaving ground, jump impulse
        is_take = np.zeros(len(tk), dtype=bool)
        is_take[1:] = (gr[:-1] == 1) & (gr[1:] == 0) & (np.diff(tk) == 1) & (v[1:] > VZ_JUMP)
        # flights: takeoff -> next grounded (exclusive)
        flight = np.zeros(len(tk), dtype=bool)
        starts = np.where(is_take)[0]
        if len(starts):
            ground = np.where(gr == 1)[0]
            ends = np.searchsorted(ground, starts)
            for s, e in zip(starts, ends):
                f = ground[e] if e < len(ground) else len(tk)
                flight[s:f] = True
        if not flight.any():
            done += len(idxs)
            continue
        starts = np.where(is_take)[0]
        take_ticks = set(tk[starts].tolist())
        if not flight.any() and not take_ticks:
            done += len(idxs)
            continue
        for i in idxs:
            T = int(Ks[i][2])
            # takeoff inside warmup [T-66, T): drop row wholesale (only
            # windows fully before a jump train; post-takeoff hidden state
            # is jump-encoded and must not supervise standard motion)
            if any((T - W <= tt < T) for tt in take_ticks):
                JM[i] = False
                JH132[i] = False
                n_rows_masked += 1
                continue
            jT = pos.get(T, -1)
            if jT < 0:
                JM[i] = False
                JH132[i] = False
                n_rows_masked += 1
                continue
            seg = flight[jT:jT + W]
            if len(seg) < W:
                pad = np.ones(W, dtype=bool)
                pad[:len(seg)] = seg
                seg = pad
            JM[i] = ~seg
            seg132 = flight[jT:jT + 132]
            if len(seg132) < 132:
                pad = np.ones(132, dtype=bool)
                pad[:len(seg132)] = seg132
                seg132 = pad
            JH132[i] = ~seg132
            n_ticks_masked += seg.sum()
        done += len(idxs)
        if done % 200000 < len(idxs):
            print('%s rows %d/%d masked-ticks=%.4f' % (
                tag, done, N, n_ticks_masked / max(done * W, 1)), flush=True)
    np.savez_compressed(os.path.join(D, tag + 'B_jumpmask.npz'), JM=JM, JH132=JH132)
    print(tag, JM.shape, 'rows-fully-masked=%d tick-mask-frac=%.4f' % (
        (JM.sum(axis=1) == 0).sum(), 1 - JM.mean()), flush=True)


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
