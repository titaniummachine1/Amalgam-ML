"""bsp_friction.py — extract per-surface friction from the BSP SURF lump.

LUMP_SURFACE = 13. Each entry is 12 bytes: uint16 friction, uint16 flags,
int32 texname id. Surfedges (lump 14) reference surfaces as int32 indices; face
texinfo (lump 13-adjacent, actually texinfo) maps faces to surfaces. For the
movement sim we only need: what friction is the player standing on. Practical
approach: build a surface->friction table and a face->surface map, then let the
caller query by point via the existing brush soup.

Usage: python ml/bsp_friction.py <bsp> -> prints the friction histogram.
"""
import struct
import sys
from collections import Counter

LUMP_SURFACE = 13
LUMP_SURFEDGES = 14
LUMP_TEXINFO = 6


def read_lump(bsp, name):
    """lump_layout entries are flat tuples of strings; find the one whose last
    field names the lump, then read it via the parsed lump objects."""
    for entry in bsp.lump_layout:
        if entry[-1] == name:
            l = entry[-2]
            fo = getattr(l, 'fileofs', None)
            fl = getattr(l, 'filelen', None)
            if fo is None or fl is None or fl == 0 or fo < 0:
                return None
            with open(bsp.filename, 'rb') as fh:
                fh.seek(fo)
                return fh.read(fl)
    return None


def main():
    from srctools.bsp import BSP
    path = sys.argv[1]
    b = BSP(path)
    surf = read_lump(b, 'surf')
    if surf is None or len(surf) % 12 != 0:
        print('no SURF lump (len=%s)' % (None if surf is None else len(surf)))
        return
    n = len(surf) // 12
    fr = []
    for i in range(n):
        f, flags, tex = struct.unpack_from('<HHi', surf, i * 12)
        fr.append(f)
    c = Counter(fr)
    print('surfaces=%d  friction values: %s' % (n, dict(c.most_common(12))))
    print('distinct friction: %s' % sorted(set(fr)))


if __name__ == '__main__':
    main()