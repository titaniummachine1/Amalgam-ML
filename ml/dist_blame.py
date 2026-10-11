import json
import os

import numpy as np

TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
seen = {}
with open(os.path.join(TMP, 'blame_rows.jsonl')) as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = (r['row'], r['h'])
        if k not in seen or seen[k][0] < r['blame']:
            seen[k] = (r['blame'], r['aneed'])
print('unique (row,h):', len(seen))
bl = np.array([v[0] for v in seen.values()])
an = np.array([v[1] for v in seen.values()])
print('blame p50/p90/p99/p99.9/max:', np.percentile(bl, [50, 90, 99, 99.9]).round(0), bl.max())
print('aneed p50/p90/p99/p99.9/max:', np.percentile(an, [50, 90, 99, 99.9]).round(0), an.max())
print('frac aneed>4000:', (an > 4000).mean(), ' >3360:', (an > 3360).mean(),
      ' >10000:', (an > 10000).mean(), ' >100000:', (an > 100000).mean())
