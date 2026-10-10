"""Repack soldier rows into contiguous raw float32 files (one-time).
Problem: Xb (trainB X, 12GB) is all-class; soldier rows scatter across it, so
per-batch fancy reads page-fault ~1.4s/batch (HDD-class latency). After repack,
block reads are sequential (~50ms). TW is already soldier-contiguous (skipped).
Outputs: trainB_X20sol.raw (N,66,20 f32), trainB_HD66.raw (N,66,2 f32),
trainB_HD132.raw (N,132,2 f32), trainB_VD66.npy, trainB_VD132.npy (bool),
trainB_S0.npy (N,5 f32). Sidecar .done = rows completed (resume-safe prefix).
Usage: python ml/repack20.py [train|eval]
"""
import json
import os
import sys
import time

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TAG = sys.argv[1] if len(sys.argv) > 1 else 'train'
BSUF = 'B'
BLK = 4096

k = np.load(os.path.join(D, TAG + 'Bk.npz'))['K']
spos = np.where(k[:, 3] == 3)[0]
N = len(spos)
print('%s soldier rows: %d' % (TAG, N), flush=True)

Xb = np.load(os.path.join(D, TAG + 'B.npz'), mmap_mode='r')['X']
TW = np.load(os.path.join(D, TAG + 'B_threatwin.npz'), mmap_mode='r')['TW']
H66 = np.load(os.path.join(D, TAG + 'B_dense66.npz'), mmap_mode='r')
H132 = np.load(os.path.join(D, TAG + 'B_dense132.npz'), mmap_mode='r')
assert TW.shape[0] == N == H66['Hd'].shape[0] == H132['Hd'].shape[0], (
    TW.shape, N, H66['Hd'].shape, H132['Hd'].shape)

P = os.path.join(D, TAG + 'B_X20sol')
fx = np.memmap(P + '.raw', dtype=np.float32, mode='w+' if not os.path.exists(P + '.raw') else 'r+',
               shape=(N, 66, 20))
fh66 = np.memmap(P.replace('X20sol', 'HD66') + '.raw', dtype=np.float32,
                 mode='w+' if not os.path.exists(P.replace('X20sol', 'HD66') + '.raw') else 'r+',
                 shape=(N, 66, 2))
fh132 = np.memmap(P.replace('X20sol', 'HD132') + '.raw', dtype=np.float32,
                  mode='w+' if not os.path.exists(P.replace('X20sol', 'HD132') + '.raw') else 'r+',
                  shape=(N, 132, 2))
done_path = P + '.done.json'
done = 0
if os.path.exists(done_path):
    done = json.load(open(done_path)).get('done', 0)
print('resuming at row %d' % done, flush=True)
t0 = time.time()
nb = 0
for a in range(done, N, BLK):
    bb = min(a + BLK, N)
    rows = spos[a:bb]
    fx[a:bb] = np.concatenate([np.array(Xb[rows]), np.array(TW[a:bb])], axis=2)
    fh66[a:bb] = np.array(H66['Hd'][a:bb])
    fh132[a:bb] = np.array(H132['Hd'][a:bb])
    fx.flush()
    del rows
    nb += 1
    if nb % 25 == 0:
        el = time.time() - t0
        rate = (a + BLK - done) / max(el, 1)
        eta = (N - a - BLK) / max(rate, 1)
        print('rows %d/%d %.1f%% %.0f rows/s eta %.0fmin' % (
            bb, N, 100.0 * bb / N, rate, eta / 60), flush=True)
    json.dump({'done': bb}, open(done_path, 'w'))
fx.flush()
fh66.flush()
fh132.flush()
print('repack done', flush=True)
vd66 = np.array(H66['Vd'])
vd132 = np.array(H132['Vd'])
np.save(P.replace('X20sol', 'VD66') + '.npy', vd66)
np.save(P.replace('X20sol', 'VD132') + '.npy', vd132)
ts = np.load(os.path.join(D, TAG + 'B_threatwin.npz'), mmap_mode='r')['TS']
np.save(P.replace('X20sol', 'S0') + '.npy', np.array(ts[:, -1, :]))
print('sidecars done', flush=True)
