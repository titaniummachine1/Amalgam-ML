"""Trainability weights: train the policy ONLY on trainable ticks.

w2 (N,66) float32 PER-TICK, multiplies the validity mask in forces_loss.
  base 1.0 everywhere valid.
  x0   TRASH (weight 0):
       - impulse-cap violations with UNKNOWN force (|A| > cap and has=0).
         Knockback WITH known force stays (has=1) - the forces rollout owns it.
       - THE BLAST TICK ITSELF (has=1 at that tick): the impulse is
         instantaneous and unpredictable - never penalize it, never train on
         it. The rollout still integrates THROUGH it (physics applies the
         impulse); only the loss comparison is skipped. Train instead on the
         recovery (below): post-blast state in, input prediction out.
  x2   recovery boost: ticks +1..+20 after a blast (knockback recovery, air
       control after RJ, landing) - the most valuable behavior per tick.
  x2   damage-reaction boost: within +-20 ticks of damage TAKEN (dodge onsets).
  x2   reversal boost: strong tangential accel (strafe-reversal proxy).
  x0.5 coast-solved downweight: coast error at 300ms already inside the hitbox.
Clamp total to [0, 4]. Eval untouched (weights are train-only).
Run AFTER ingest AND build_knock (needs KB). Output: trainB_w2.npz.
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
SUF = os.environ.get('OUT_SUFFIX', 'B')
DT = 1 / 66.0
DID2CSV = _dr.reverse(_dr.load())


def build(tag):
    k = np.load(os.path.join(DS, tag + SUF + 'k.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    N = len(Ks)
    d = np.load(os.path.join(DS, tag + SUF + '.npz'), mmap_mode='r')
    Y = np.array(d['Y'])[sol]
    X = np.array(d['X'][sol])
    H = np.array(d['H'])[sol]
    dn = np.load(os.path.join(DS, tag + SUF + '_dense66.npz'))
    Vd = np.array(dn['Vd'])
    assert Vd.shape[0] == N, (Vd.shape, N)
    kb = None
    kp = os.path.join(DS, tag + SUF + '_knock.npz')
    if os.path.exists(kp):
        kb = np.load(kp)['KB']
        assert kb.shape[0] == N, (kb.shape, N)
    w2 = np.ones((N, 66), dtype=np.float32)
    # --- trash: impulse-cap violations with UNKNOWN force ---
    amag = np.linalg.norm(Y, axis=1) * 240.0
    # cap per sample: ground 14*MAXV else 1500 (builder rule); unknown force only
    gr = X[:, -1, 3] > 0.5
    cap = np.where(gr, 14.0 * 240.0, 1500.0)
    viol = amag > cap
    if kb is not None:
        has = kb[:, :, 3].max(axis=1) > 0.5  # any known force in window
        trash = viol & ~has
    else:
        trash = viol
    # trash zeroes the whole sample (label tick unlearnable); per-tick refinement
    # comes from the dense validity mask already. Mark for reporting:
    print('%s impulse-cap violations: %.4f (%.4f with known force -> kept)' % (
        tag, viol.mean(),
        (viol & (kb[:, :, 3].max(axis=1) > 0.5)).mean() if kb is not None else 0.0),
        flush=True)
    # --- damage-reaction boost (+-20 ticks of damage taken) ---
    dmg_ticks = defaultdict(set)
    for did in sorted(set(int(r[0]) for r in Ks)):
        base = DID2CSV.get(did)
        if base is None:
            continue
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
                if row.get('kind') != 'hurt':
                    continue
                try:
                    t, dmgv, posthp = int(row['tick']), int(row['c']), int(row['e'])
                except ValueError:
                    continue
                for ent, hs in health.items():
                    hit = False
                    for dt in range(0, 4):
                        h = hs.get(t + dt)
                        if h is None:
                            continue
                        if h == posthp or (hs.get(t + dt - 1) is not None
                                           and hs[t + dt - 1] - h >= max(dmgv - 15, 1)):
                            hit = True
                            break
                    if hit:
                        for dt in range(-20, 21):
                            dmg_ticks[(did, ent)].add(t + dt)
    n_dmg = 0
    for i in range(N):
        s = dmg_ticks.get((int(Ks[i][0]), int(Ks[i][1])))
        if not s:
            continue
        if int(Ks[i][2]) in s:
            w2[i] *= 2.0
            n_dmg += 1
    print('%s label ticks within +-20 of damage taken: %d (%.4f)' % (
        tag, n_dmg, n_dmg / N), flush=True)
    # --- blast-tick exclusion + recovery boost (per-tick, from KB) ---
    # The blast frame is unpredictable: weight 0 at has=1 ticks, x2 for the
    # +1..+20 recovery ticks after each blast. Recovery is the learnable part.
    kp = os.path.join(DS, tag + SUF + '_knock.npz')
    n_blast = n_rec = 0
    if os.path.exists(kp):
        has = np.array(np.load(kp)['KB'][:, :, 3] > 0.5)
        assert has.shape[0] == N, (has.shape, N)
        w2[has] = 0.0
        n_blast = int(has.sum())
        rec = np.zeros_like(has, dtype=bool)
        for d in range(1, 21):
            rec[:, d:] |= has[:, :-d]
        rec[has] = False  # blast tick itself stays 0, never boosted
        w2[rec] *= 2.0
        n_rec = int(rec.sum())
    print('%s blast ticks excluded: %d  recovery ticks boosted: %d' % (
        tag, n_blast, n_rec), flush=True)
    # --- reversal boost: lateral accel sign flip (labels, ego frame) ---
    # tangential = accel cross velocity-dir at label tick
    v = X[:, -1, :2] * 240.0
    sp = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / np.maximum(sp, 1e-6)
    tang = -Y[:, 1] * u[:, 0] + Y[:, 0] * u[:, 1]
    rad = Y[:, 0] * u[:, 0] + Y[:, 1] * u[:, 1]
    # reversal = strong tangential accel (strafe authority); proxy for flips
    rev = (np.abs(tang) > 2.0) & (sp[:, 0] > 50)
    w2[rev] *= 2.0
    print('%s strong-tangential (reversal-proxy) labels: %.4f' % (tag, rev.mean()),
          flush=True)
    # --- coast-solved downweight: coast already inside hitbox at 300ms ---
    v0 = X[:, -1, :2] * 240.0
    tgt = H[:, 2]  # 300ms horizon (HOR index 2 = 20 ticks)
    mm = ~np.isnan(tgt[:, 0])
    coast_err = np.full(N, np.inf)
    coast_err[mm] = np.abs(v0[mm] * (20 * DT) - tgt[mm]).max(axis=1)
    solved = mm & (coast_err < 24.0)
    w2[solved] *= 0.5
    print('%s coast-solved at 300ms (downweighted): %.4f' % (tag, solved.mean()),
          flush=True)
    # trash samples -> 0 (impulse-cap violations, whole-sample backstop)
    w2[trash] = 0.0
    # a blast tick zeroes ONLY its own tick (above); but a LABEL tick whose own
    # accel violates the cap with unknown force is unlearnable -> zero its row.
    # (Known-force rows stay: the forces rollout owns them.)
    np.clip(w2, 0.0, 4.0, out=w2)
    np.savez_compressed(os.path.join(DS, tag + SUF + '_w2.npz'), W2=w2)
    print('%s W2: mean %.3f frac0 %.4f frac>1 %.4f' % (
        tag, w2.mean(), (w2 == 0).mean(), (w2 > 1).mean()), flush=True)


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
