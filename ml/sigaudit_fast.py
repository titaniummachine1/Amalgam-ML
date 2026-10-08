"""Fast signature audit: C-speed literal-run search per pattern.
For each pattern, take the longest literal run, bytes.find() it (C memchr),
then verify full pattern at each candidate. Reports zero-hit patterns.
Usage: python ml/sigaudit_fast.py"""
import glob
import os
import re

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
    import struct
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
    return [None if set(t) <= set("?") else int(t, 16) for t in pat.split()]


def count_fast(blob, pat):
    # longest literal run -> find -> verify
    best, bi = b"", 0
    cur, ci = bytearray(), 0
    for i, b in enumerate(pat):
        if b is None:
            if len(cur) > len(best):
                best, bi = bytes(cur), ci
            cur, ci = bytearray(), i + 1
        else:
            if not cur:
                ci = i
            cur.append(b)
    if len(cur) > len(best):
        best, bi = bytes(cur), ci
    if not best:
        return -1
    n = 0
    start = 0
    while True:
        j = blob.find(best, start)
        if j < 0:
            break
        o = j - bi
        if o >= 0 and all(b is None or blob[o + k] == b for k, b in enumerate(pat)):
            n += 1
        start = j + 1
    return n


def main():
    sigs = []
    for f in glob.glob(os.path.join(SRC, "**", "*.cpp"), recursive=True):
        try:
            t = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in re.finditer(r'MAKE_SIGNATURE\(\s*(\w+)\s*,\s*"([\w.]+)"\s*,\s*"([0-9A-Fa-f? ]+)"', t):
            sigs.append(m.groups())
    print("signatures: %d" % len(sigs), flush=True)
    blobs = {}
    for mod, path in MODDIR.items():
        blobs[mod] = read_text(path) if os.path.exists(path) else None
        print("%s %s" % (mod, len(blobs[mod]) if blobs[mod] else "MISSING"), flush=True)
    bad = {}
    for name, mod, pat in sigs:
        b = blobs.get(mod.lower())
        if b is None:
            bad.setdefault(mod, []).append((name, "no dll"))
            continue
        c = count_fast(b, parse(pat))
        if c == 0:
            bad.setdefault(mod, []).append((name, pat))
    print("\n==== STALE ====", flush=True)
    tot = 0
    for mod, lst in sorted(bad.items()):
        print("-- %s: %d" % (mod, len(lst)), flush=True)
        for name, pat in lst:
            print("   %s" % name, flush=True)
            tot += 1
    print("total stale / total: %d / %d" % (tot, len(sigs)), flush=True)


if __name__ == "__main__":
    main()
