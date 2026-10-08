"""Append-only demo-id registry. did values are STABLE FOREVER once assigned.
Bootstrapped: original 25 tfmge_ CSVs (sorted) -> 0..24 (verified vs DB 2026-09-30).
New files appended in sorted order -> 25+."""
import json
import os

DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
REG = os.path.join(DS, 'did_registry.json')

_BOOT = [
 'tfmge_2026-09-27__13-35_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-27__14-35_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-27__15-51_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-27__16-51_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-27__18-10_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-27__19-34_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-27__20-03_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-27__21-03_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-27__22-43_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-27__23-43_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__00-43_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__15-39_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__16-39_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__16-56_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__16-58_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__17-18_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__17-20_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__18-20_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__19-28_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__19-29_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__20-55_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__21-42_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__22-02_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-28__23-02_mge_training_v8_beta4b.dem.csv',
 'tfmge_2026-09-29__00-02_mge_training_v8_beta4b.dem.csv',
]


def load():
    if os.path.exists(REG):
        with open(REG) as f:
            return json.load(f)
    reg = {name: i for i, name in enumerate(_BOOT)}
    save(reg)
    return reg


def save(reg):
    with open(REG, 'w') as f:
        json.dump(reg, f, indent=1)


def register(names):
    """Assign fresh dids to unseen basenames (sorted). Returns full registry."""
    reg = load()
    nxt = max(reg.values()) + 1 if reg else 0
    for name in sorted(set(names) - set(reg)):
        reg[name] = nxt
        nxt += 1
    save(reg)
    return reg


def reverse(reg):
    return {v: k for k, v in reg.items()}
