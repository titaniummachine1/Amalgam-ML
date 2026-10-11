"""Per-tick input extraction (wishdir) from inverse dynamics, engine-exact.
For each soldier row: true velocities from Hd132 diffs, friction-cancelled
dv = v_true - friction(v_prev) (grounded AND air: air = friction-identity,
same formula, no special cases). wishdir = dv/|dv|, wishspeed tau =
clamp(|dv|/(ACCEL*DT*wcap)) with wcap = class MAXV (240 soldier).
(fm,sm) = wishdir * tau (matches inputspace policy_out radial, tau<=1).
Confidence IM=0 where: invalid targets, jump-flight ticks (JH132), unnatural
ticks (|a_needed| > caps: 4000 ground / 3000 air), or |dv| ~ 0 (coast:
input unobservable when addspeed<=0). Output trainB_inputs.npz / 
evalB_inputs.npz {IN (N,132,2) float32, IM (N,132) bool}, soldier-ordered.
Future use: per-tick input-aux supervision (the supervision inputspace died
without). Chunked vectorized, no DB reads (Hd repack + X20 only).
Usage: python ml/extract_inputs.py [train|eval|both]"""
import os
import sys

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DT = 1 / 66.0
ACCEL = 10.0
MAXV = 240.0
CAP_GR, CAP_AIR = 4000.0, 3000.0
CHUNK = 20000


def friction_np(v, gr):
    sp = np.linalg.norm(v, axis=-1, keepdims=True)
    drop = np.maximum(sp, 100.0) * 4.0 * DT
    newsp = np.maximum(sp - drop, 0.0)
    scale = np.divide(newsp, sp, out=np.ones_like(newsp), where=sp > 0)
    return v * (gr * scale + (1.0 - gr))


def build(tag):
    P = os.path.join(D, tag + 'B_X20sol')
    n = int(os.path.getsize(P + '.raw') // (66 * 20 * 4))
    X = np.memmap(P + '.raw', dtype=np.float32, mode='r', shape=(n, 66, 20))
    H = np.memmap(P.replace('X20sol', 'HD132') + '.raw', dtype=np.float32, mode='r',
                  shape=(n, 132, 2))
    V = np.load(P.replace('X20sol', 'VD132') + '.npy', mmap_mode='r')
    JH = None
    if tag == 'train':
        JH = np.load(os.path.join(D, 'trainB_jumpmask.npz'))['JH132']
    IN = np.zeros((n, 132, 2), dtype=np.float32)
    IM = np.zeros((n, 132), dtype=bool)
    for a in range(0, n, CHUNK):
        bb = min(a + CHUNK, n)
        xb = np.array(X[a:bb])
        Hd = np.array(H[a:bb])
        Vd = np.array(V[a:bb])
        v0 = xb[:, -1, :2] * MAXV
        gr = xb[:, -1, 3:4]
        Hd0 = np.concatenate([np.zeros_like(Hd[:, :1]), Hd], axis=1)
        Vt = (Hd0[:, 1:] - Hd0[:, :-1]) / DT
        Vf = friction_np(np.concatenate([v0[:, None, :], Vt[:, :-1]], axis=1),
                         gr[:, None, :])
        dv = Vt - Vf
        sp = np.linalg.norm(dv, axis=-1, keepdims=True)
        Aneed = np.abs(dv).max(-1)
        unnat = Aneed > np.where(gr > 0.5, CAP_GR, CAP_AIR)
        tau = np.minimum(sp.squeeze(-1) / (ACCEL * DT * MAXV), 1.0)
        wh = np.divide(dv, sp, out=np.zeros_like(dv), where=sp > 1e-6)
        conf = (Vd & (sp.squeeze(-1) > 8.0) & (~unnat)).astype(bool)
        if JH is not None:
            conf = conf & JH[a:bb]
        IN[a:bb] = (wh * tau[..., None]).astype(np.float32)
        IM[a:bb] = conf
        if (a // CHUNK) % 5 == 0:
            print('%s rows %d/%d conf-frac=%.4f' % (tag, bb, n, IM[:bb].mean()), flush=True)
    np.savez_compressed(os.path.join(D, tag + 'B_inputs.npz'),
                        IN=IN, IM=IM)
    print(tag, IN.shape, 'confident frac=%.4f' % IM.mean(), flush=True)


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'both'
    for t in (['train', 'eval'] if arg == 'both' else [arg]):
        build(t)
