"""build_all_rollouts.py — corpus driver: per-CSV rollout npz + train/eval split.
Usage: python ml/build_all_rollouts.py <csv_list.txt> <outdir> <soup.npz> <stride> <eval_every> [chunk_idx chunks]
  chunk_idx/chunks: process only this slice (parallel workers).
Skips outputs that exist and load (resume-safe). Logs per-CSV outcome.
Eval = every eval_every-th CSV (by demo, no leakage).
"""
import subprocess
import sys

import numpy as np


def main():
    lst, outdir, soup, stride, ev = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    ci = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    nc = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    import os
    os.makedirs(outdir, exist_ok=True)
    with open(lst) as fh:
        csvs = [l.strip() for l in fh if l.strip()]
    mine = [c for i, c in enumerate(csvs) if i % nc == ci]
    print('chunk %d/%d: %d/%d csvs' % (ci, nc, len(mine), len(csvs)), flush=True)
    tr, evl = [], []
    for n, csv in enumerate(mine):
        base = __import__('os').path.basename(csv)
        stem = base[:-4] if base.endswith('.csv') else base
        out = __import__('os').path.join(outdir, 'rollout_' + stem + '.npz')
        ok = False
        if __import__('os').path.exists(out) and __import__('os').path.getsize(out) > 10000:
            try:
                d = np.load(out, allow_pickle=True)
                ok = 'X' in d.files  # loads = done (duds save empty rows)
            except Exception:
                ok = False
        if not ok:
            r = subprocess.run([sys.executable, 'ml/build_rollout.py', csv, soup, out, stride],
                               capture_output=True, text=True, cwd='C:/gitProjects/amalgam')
            tail = (r.stdout + r.stderr)[-300:].replace('\n', ' | ')
            print('[%d/%d] %s -> %s' % (n + 1, len(mine), base[:40], tail[-160:]), flush=True)
            if r.returncode != 0:
                continue
        (evl if (csvs.index(csv) % ev == 0) else tr).append(out)
    with open(__import__('os').path.join(outdir, 'train_list_c%d.txt' % ci), 'w') as fh:
        fh.write('\n'.join(tr))
    with open(__import__('os').path.join(outdir, 'eval_list_c%d.txt' % ci), 'w') as fh:
        fh.write('\n'.join(evl))
    print('chunk %d done: train=%d eval=%d' % (ci, len(tr), len(evl)), flush=True)


main()
