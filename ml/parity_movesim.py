"""parity_movesim.py — differential test: movesim.py vs Rust rollout-cli.
20k seeded random cases across ground/air/inverse. PASS if max abs diff < 1e-9.
Usage: python ml/parity_movesim.py [n]
Build the binary first: cargo build -p rollout-cli --release (in tf2-movesim).
"""
import math
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from movesim import (air_accelerate, ground_step, inverse_ground)
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
BIN = r'C:\gitProjects\tf2-movesim\target\release\rollout-cli.exe'
rng = random.Random(1337)
lines = []
expect = []
for _ in range(N):
    k = rng.randrange(3)
    if k == 0:
        v = [rng.uniform(-600, 600), rng.uniform(-600, 600)]
        a = rng.uniform(0, 2 * math.pi)
        f = [math.cos(a), math.sin(a)]
        ws = rng.uniform(0, 400)
        mx = rng.choice([230.0, 240.0, 280.0, 300.0, 320.0, 400.0])
        lines.append('0 %.17g %.17g %.17g %.17g %.17g %.17g' % (v[0], v[1], f[0], f[1], ws, mx))
        u, s = ground_step(np.array([v[0], v[1], 0.0]), np.array(f), ws, mx)
        expect.append((u[0], u[1], u[2], float(s)))
    elif k == 1:
        v = [rng.uniform(-600, 600), rng.uniform(-600, 600), rng.uniform(-600, 600)]
        a = rng.uniform(0, 2 * math.pi)
        f = [math.cos(a), math.sin(a)]
        ws = rng.uniform(0, 400)
        lines.append('1 %.17g %.17g %.17g %.17g %.17g %.17g 30.0' % (v[0], v[1], v[2], f[0], f[1], ws))
        u = air_accelerate(np.array(v), np.array([f[0], f[1], 0.0]), ws, 30.0)
        expect.append((u[0], u[1], u[2]))
    else:
        v0 = [rng.uniform(-600, 600), rng.uniform(-600, 600)]
        v1 = [rng.uniform(-600, 600), rng.uniform(-600, 600)]
        mx = 300.0
        lines.append('2 %.17g %.17g %.17g %.17g %.17g' % (v0[0], v0[1], v1[0], v1[1], mx))
        r = inverse_ground(np.array([v0[0], v0[1], 0.0]),
                           np.array([v1[0], v1[1], 0.0]), mx)
        wd = r['wishdir']
        info = r['info']
        ri = 2 if info['capped'] else (1 if info['boosted'] else 0)
        if wd is None:
            expect.append(('nan', 'nan', r['wishspeed'], float(ri)))
        else:
            expect.append((wd[0], wd[1], r['wishspeed'], float(ri)))
p = subprocess.run([BIN], input='\n'.join(lines) + '\n', capture_output=True,
                   text=True)
assert p.returncode == 0, p.stderr[-500:]
got = [l.split() for l in p.stdout.strip().split('\n')]
assert len(got) == len(expect), (len(got), len(expect))
worst = 0.0
for e, g in zip(expect, got):
    for a, b in zip(e, g):
        if isinstance(a, str) and a == 'nan':
            assert b.lower() == 'nan', (a, b)
            continue
        d = abs(float(a) - float(b))
        worst = max(worst, d)
print('cases=%d worst_abs_diff=%.3e GATE=%s' % (
    N, worst, 'PASS' if worst < 1e-9 else 'FAIL'))
sys.exit(0 if worst < 1e-9 else 1)
