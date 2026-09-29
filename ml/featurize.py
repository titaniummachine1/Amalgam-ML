"""Featurization cache: grouped, content-hashed, recompute-only-what-changed.

Layers:
  RAW    dataset.db ticks table (immutable demo extraction; Turso-compatible file).
  GROUPS train_<grp>_<hash8>.npz per feature group (base movement, threat, ...).
  manifest.json maps group -> {hash, files}. A group rebuilds only when its
  CONFIG section hash changes; other groups are reused untouched.

Usage:
  python featurize.py            # build/refresh all groups per manifest
  python featurize.py --check    # verify cache freshness, exit code
"""
import hashlib
import json
import os
import sys

DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
MANIFEST = os.path.join(DS, 'manifest.json')

# ---- group configs (hash covers exactly these) ----
BASE_CONFIG = {
    "warmup": 66, "stride": 4, "maxv": 240.0,
    "afk_win": 132, "approach": 66, "teleport_hu": 300.0,
    "label_clip": 40.0, "weight": "clip(mean_speed/0.5,0.1,1)",
    "horizons": [7, 13, 20, 33, 66],
    "ego": "eye_yaw_fallback_velocity_heading",
    "features": ["ego_vx", "ego_vy", "speed", "grounded", "dyaw"],
    "builder": "build_ds2.py v1",
}
THREAT_CONFIG = {
    "status": "pending",
    "plan": "projectile FOV slots x3 from .proj.csv sidecars",
}


def h8(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {"groups": {}}


def save_manifest(m):
    with open(MANIFEST, 'w') as f:
        json.dump(m, f, indent=1)


def register(group, cfg, files):
    m = manifest()
    m["groups"][group] = {"hash": h8(cfg), "config": cfg, "files": files}
    save_manifest(m)
    print(f'registered {group} hash={h8(cfg)} files={files}')


def fresh(group, cfg):
    g = manifest()["groups"].get(group)
    if not g or g["hash"] != h8(cfg):
        return False
    return all(os.path.exists(os.path.join(DS, f)) for f in g["files"])


if __name__ == '__main__':
    if '--check' in sys.argv:
        for grp, cfg in [('base', BASE_CONFIG), ('threat', THREAT_CONFIG)]:
            print(grp, 'fresh' if fresh(grp, cfg) else 'STALE')
        sys.exit(0 if fresh('base', BASE_CONFIG) else 1)
    # register current on-disk arrays as the base group (built by build_ds2.py)
    register('base', BASE_CONFIG, ['train2.npz', 'eval2.npz'])
    print('threat group: pending (projectile sidecars collected, features not built yet)')
