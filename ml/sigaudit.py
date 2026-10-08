"""Audit every MAKE_SIGNATURE in the tree against current game DLLs.
Reports patterns with ZERO hits (stale/broken) grouped by module.
Usage: python ml/sigaudit.py  (run from anywhere; paths hardcoded)."""
import glob
import os
import re
import struct
import sys

SRC = r"C:\gitProjects\amalgam\Amalgam\src"
MODDIR = {
    "engine.dll": r"C:\Steam\steamapps\common\Team Fortress 2\bin\x64\engine.dll",
    "client.dll": r"C:\Steam\steamapps\common\Team Fortress 2\tf\bin\x64\client.dll",
    "server.dll": r"C:\Steam\steamapps\common\Team Fortress 2\tf\bin\x64\server.dll",
    "materialsystem.dll": r"C:\Steam\steamapps\common\Team Fortress 2\bin\x64\materialsystem.dll",
    "vphysics.dll": r"C:\Steam\steamapps\common\Team Fortress 2\bin\x64\vphysics.dll",
    "vgui2.dll": r"C:\Steam\steamapps\common\Team Fortress 2\bin\x64\vgui2.dll",
}


def read_text(path):
    with open(path, "rb") as f:
        data = f.read()
    e = struct.unpack("<I", data[0x3C:0x40])[0]
    nsec = struct.unpack("<H", data[e + 6:e + 8])[0]
    se = e + 24 + struct.unpack("<H", data[e + 20:e + 22])[0]
    for i in range(nsec):
        off = se + i * 40
        if data[off:off + 8].rstrip(b"\x00") == b".text":
            vsz, vaddr, rawsz, rawptr = struct.unpack("<IIII", data[off + 8:off + 24])
            return data[rawptr:rawptr + rawsz]
    return None


def parse(pat):
    out = []
    for tok in pat.split():
        out.append(None if set(tok) <= set("?") else int(tok, 16))
    return out


def count(blob, pat):
    n, c = len(pat), 0
    for i in range(len(blob) - n):
        for j, b in enumerate(pat):
            if b is not None and blob[i + j] != b:
                break
        else:
            c += 1
    return c


def main():
    sigs = []
    for f in glob.glob(os.path.join(SRC, "**", "*.cpp"), recursive=True):
        try:
            t = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in re.finditer(r'MAKE_SIGNATURE\(\s*(\w+)\s*,\s*"([\w.]+)"\s*,\s*"([0-9A-Fa-f? ]+)"', t):
            sigs.append(m.groups())
    print("signatures found: %d" % len(sigs), flush=True)
    blobs = {}
    for mod, path in MODDIR.items():
        if os.path.exists(path):
            blobs[mod] = read_text(path)
            print("%s .text %d bytes" % (mod, len(blobs[mod])), flush=True)
        else:
            print("%s MISSING FILE" % mod, flush=True)
    bad = {}
    for name, mod, pat in sigs:
        b = blobs.get(mod.lower())
        if b is None:
            bad.setdefault(mod, []).append((name, "no dll"))
            continue
        if count(b, parse(pat)) == 0:
            bad.setdefault(mod, []).append((name, pat))
    print("\n==== STALE (zero hits) ====", flush=True)
    tot = 0
    for mod, lst in sorted(bad.items()):
        print("-- %s: %d stale" % (mod, len(lst)), flush=True)
        for name, pat in lst:
            print("   %s : %s" % (name, pat), flush=True)
            tot += 1
    print("total stale: %d / %d" % (tot, len(sigs)), flush=True)


if __name__ == "__main__":
    main()
