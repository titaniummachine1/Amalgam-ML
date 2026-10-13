"""demoin.py — true usercmd extractor for HLTV (SourceTV) demos.

Source-perspective STV demos record every client's CLC_Move, so the REAL
per-tick inputs (buttons / forwardmove / sidemove / viewangles) are on disk.
That replaces noisy inverse-recovery with exact packets.

Format is taken from the leak (read-only), not guessed:
  engine/demofile.cpp:143-194  WriteCmdHeader/ReadCmdHeader -> u8 cmd, i32 tick
  engine/demofile.cpp:152-163  cmd names: 1=dem_signon 2=dem_packet 5=dem_usercmd
                               6=dem_datatables 7=dem_stop 8=dem_stringtables
  demoheader_t                1072 bytes (8+4+4+260*4+4+4+4+4)
  common/netmessages.cpp:86-95 CLC_Move::ReadFromBuffer
        nNewCommands(4b) nBackupCommands(3b) u16 length, then `length` bits
  game/shared/usercmd.cpp:199-306 ReadUsercmd (delta bitmask, fields below)
  common/protocol.h           NUM_NEW_COMMAND_BITS 4, NUM_BACKUP_COMMAND_BITS 3
  engine/net.h:73             NETMSG_TYPE_BITS 6
  ehta.h                      MAX_EDICT_BITS 11

Frames: playback_frames == playback_ticks (1 frame/tick) so the frame header
(cu8 cmd=dem_packet + i32 tick) is found by its sequential-tick signature; that
removes any need to guess payload length.
"""
import struct
import sys

# message ids — common/protocol.h:86-161 (authoritative, from leak)
NETMSG_NOP = 0
NETMSG_Disconnect = 1
NETMSG_File = 2
NETMSG_Tick = 3
NETMSG_StringCmd = 4
NETMSG_SetConVar = 5
NETMSG_SignonState = 6
svc_Print = 7
svc_ServerInfo = 8
svc_SendTable = 9
svc_ClassInfo = 10
svc_SetPause = 11
svc_CreateStringTable = 12
svc_UpdateStringTable = 13
svc_VoiceInit = 14
svc_VoiceData = 15
svc_Sounds = 17
svc_SetView = 18
svc_FixAngle = 19
svc_CrosshairAngle = 20
svc_BSPDecal = 21
svc_UserMessage = 23
svc_EntityMessage = 24
svc_GameEvent = 25
svc_PacketEntities = 26
svc_TempEntities = 27
svc_Prefetch = 28
svc_Menu = 29
svc_GameEventList = 30
svc_GetCvarValue = 31
svc_CmdKeyValues = 32
svc_SetPauseTimed = 33

clc_ClientInfo = 8
CLC_MOVE = 9
clc_VoiceData = 10
clc_BaselineAck = 11
clc_ListenEvents = 12
clc_RespondCvarValue = 13
clc_FileCRCCheck = 14
clc_SaveReplay = 15
clc_CmdKeyValues = 16
clc_FileMD5Check = 17

INVIS = 1 << 14
INVAR = (1 << 10) - 1
BITS = 32

MAX_EDICT_BITS = 11
NUM_NEW_COMMAND_BITS = 4
NUM_BACKUP_COMMAND_BITS = 3
NETMSG_TYPE_BITS = 6

BTN_ATTACK = 1 << 0
BTN_JUMP = 1 << 1
BTN_DUCK = 1 << 2
BTN_FORWARD = 1 << 3
BTN_BACK = 1 << 4
BTN_MOVELEFT = 1 << 5
BTN_MOVERIGHT = 1 << 6


class UCmd(object):
    __slots__ = ('command_number', 'tick_count', 'p', 'y', 'r', 'forwardmove',
                 'sidemove', 'upmove', 'buttons', 'impulse')

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, 0)


class BitReader(object):
    """bf_read equivalent (LSB-first, matches source bf_read)."""

    def __init__(self, data, bitpos=0):
        self.d = data
        self.p = bitpos

    def read(self, n):
        v = 0
        d = self.d
        p = self.p
        if p + n > len(d) * 8:
            raise IndexError('bitread overrun')
        for i in range(n):
            byte = d[p >> 3]
            bit = (byte >> (p & 7)) & 1
            v |= bit << i
            p += 1
        self.p = p
        return v

    def read_sbit(self, n):
        v = self.read(n)
        if v & (1 << (n - 1)):
            v -= (1 << n)
        return v

    def read_float(self):
        self.p = (self.p + 7) & ~7
        v = struct.unpack('<f', self.d[self.p >> 3:self.p // 8 + 4])[0]
        self.p += 32
        return v

    def read_string(self):
        out = bytearray()
        while True:
            c = self.d[self.p >> 3]
            self.p += 8
            if c == 0:
                break
            out.append(c)
        return out.decode('latin1')

    def read_bit_coord(self):
        if self.read(1):
            return float(self.read(8))
        if self.read(1):
            return float(self.read_sbit(4)) * (1.0 / 8.0)
        return float(self.read_sbit(12)) * (1.0 / 4096.0)


def read_usercmd(br, prev):
    """game/shared/usercmd.cpp:199-306. prev = previous cmd (delta base)."""
    c = UCmd()
    c.command_number = prev.command_number
    c.tick_count = prev.tick_count
    c.p, c.y, c.r = prev.p, prev.y, prev.r
    c.forwardmove, c.sidemove, c.upmove = prev.forwardmove, prev.sidemove, prev.upmove
    c.buttons, c.impulse = prev.buttons, prev.impulse
    if br.read(1):
        c.command_number = br.read(32)
    else:
        c.command_number = prev.command_number + 1
    if br.read(1):
        c.tick_count = br.read(32)
    else:
        c.tick_count = prev.tick_count + 1
    if br.read(1):
        c.p = br.read_float()
    if br.read(1):
        c.y = br.read_float()
    if br.read(1):
        c.r = br.read_float()
    if br.read(1):
        c.forwardmove = br.read_float()
    if br.read(1):
        c.sidemove = br.read_float()
    if br.read(1):
        c.upmove = br.read_float()
    if br.read(1):
        c.buttons = br.read(32)
    if br.read(1):
        c.impulse = br.read(8)
    if br.read(1):                       # weaponselect
        br.read(MAX_EDICT_BITS)
        if br.read(1):                   # weaponsubtype
            br.read(8)
    if br.read(1):                       # mousedx
        br.read(16)
    if br.read(1):                       # mousedy
        br.read(16)
    return c


FRAME_HDR = 93   # cmd(1)+tick(4)+democmdinfo_t(76)+seq(4)+seqack(4)+len(4)
CMDINFO_LEN = 76  # int flags + viewOrigin/viewAngles/localViewAngles + *2 (3 floats each)


def read_header(path):
    b = open(path, 'rb').read(1072)
    if b[:8] != b'HL2DEMO\x00':
        raise ValueError('not a demo: %r' % b[:8])
    netproto, demoproto = struct.unpack('<ii', b[8:16])
    mp = b[536:796].split(b'\x00')[0].decode('latin1')
    pt, pticks, pframes, slen = struct.unpack('<fiii', b[1056:1072])
    return {'netproto': netproto, 'demoproto': demoproto, 'map': mp,
            'ticks': pticks, 'frames': pframes, 'signonlen': slen}


def frame_offsets(b, start, nframes=None):
    """Walk dem_packet frames: cmd(1) tick(4) cmdinfo(76) seq seqack len payload.
    democmdinfo_t size verified empirically against HLTV frame stride."""
    out = []
    off = start
    end = len(b)
    while off + FRAME_HDR <= end:
        cmd = b[off]
        tick = struct.unpack('<i', b[off + 1:off + 5])[0]
        ln = struct.unpack('<i', b[off + 89:off + 93])[0]
        if cmd != 2 or not (0 <= ln <= 1 << 20):
            return out
        out.append((off, tick, ln))
        off += FRAME_HDR + ln
        if nframes and len(out) >= nframes:
            break
    return out


def load_commands(path, nframes=None, probe=False):
    """Return (header, {frame_tick: [Ucmd]}) using true CLC_Move packets."""
    b = open(path, 'rb').read()
    h = read_header(path)
    start = 1072 + 5 + h['signonlen']
    fr = frame_offsets(b, start, nframes)
    cmds = {}
    ids = {}
    for i, (off, tick, ln) in enumerate(fr):
        payload = b[off + FRAME_HDR:off + FRAME_HDR + ln]
        if not payload:
            continue
        moves, endp = parse_packet(payload, probe=probe, ids=ids)
        for cm in moves:
            if cm:
                cmds[tick] = cm
        if probe and i < 3:
            print('frame tick=%d payload=%d bytes msgs=%s consumed=%d/%d' %
                  (tick, ln, ids, endp, ln * 8))
    return h, cmds, len(fr), ids


def parse_packet(payload, probe=False, ids=None, want_move=True):
    """Walk netmessages in one frame payload; return (moves, bits_consumed)."""
    br = BitReader(payload, 0)
    nbits = len(payload) * 8
    moves = []
    n = 0
    while br.p + NETMSG_TYPE_BITS <= nbits and n < 512:
        start = br.p
        msgid = br.read(NETMSG_TYPE_BITS)
        if ids is not None:
            ids[msgid] = ids.get(msgid, 0) + 1
        if msgid >= 63:
            msgid = 63 + br.read(4)
        if msgid == CLC_MOVE:
            nnew = br.read(NUM_NEW_COMMAND_BITS)
            nback = br.read(NUM_BACKUP_COMMAND_BITS)
            length = br.read(16)
            endp = br.p + length
            if want_move and endp <= nbits:
                prev = UCmd()
                cmds = []
                for _ in range(nnew):
                    prev = read_usercmd(br, prev)
                    cmds.append(prev)
                moves.append(cmds)
            br.p = endp
            n += 1
            continue
        if not _skip_message(br, msgid, nbits):
            if probe:
                print('  stop at bit %d on msgid %d' % (start, msgid))
            return moves, start
        n += 1
    return moves, br.p


def _skip_message(br, msgid, nbits):
    """Skip one message. Returns False if unknown (caller stops).
    Layouts from common/netmessages.cpp ReadFromBuffer + protocol.h ids."""
    try:
        if msgid == NETMSG_Tick:                       # net_Tick=3
            br.read(32); br.read_float(); br.read_float()
        elif msgid == svc_ServerInfo:                  # 8
            br.read(32); br.read(32); br.read_string()
            br.read_string(); br.read_string(); br.read_string(); br.read_string()
            br.read(8); br.read(8); br.read(8)
        elif msgid == NETMSG_SignonState:              # 6
            br.read(32); br.read(32)
        elif msgid == svc_SetPause:                    # 11
            br.read(1)
        elif msgid == svc_SetPauseTimed:               # 33
            br.read_float(); br.read_float(); br.read_float()
        elif msgid == svc_ClassInfo:                   # 10
            br.read(32); br.read_string(); br.read_string()
        elif msgid == svc_CreateStringTable:           # 12
            br.read(1); br.read(16); br.read_string()
        elif msgid == svc_UpdateStringTable:           # 13
            br.read(INVAR); br.read(8)
            if br.read(1):
                n = br.read(16)
                for _ in range(n):
                    br.read_string(); br.read_string()
            else:
                n = br.read(16)
                for _ in range(n):
                    br.read(1); br.read(INVAR); br.read(1); br.read(INVAR)
        elif msgid == svc_VoiceInit:                   # 14
            br.read(8); br.read(32); br.read(1); br.read(32); br.read(32); br.read(32)
        elif msgid == svc_Sounds:                      # 17
            br.read(32); br.read(32)
            br.read(INVAR); br.read(1)
            br.read_float(); br.read_float()
        elif msgid == svc_SetView:                     # 18
            br.read(INVAR); br.read(32)
        elif msgid == svc_FixAngle:                    # 19
            br.read_float(); br.read_float()
        elif msgid == svc_CrosshairAngle:              # 20
            br.read_float()
        elif msgid == svc_BSPDecal:                    # 21
            br.read(INVAR); br.read(INVAR)
        elif msgid == svc_UserMessage:                 # 23
            br.read(32)
        elif msgid == svc_EntityMessage:               # 24
            br.read(32)
        elif msgid == svc_GameEvent:                   # 25
            br.p += br.read(16)
        elif msgid == svc_PacketEntities:              # 26
            return _skip_packet_entities(br, nbits)
        elif msgid == svc_TempEntities:                # 27
            br.read(INVAR)
            br.p += br.read(INVAR) * 48
        elif msgid == svc_Prefetch:                    # 28
            br.p += br.read(16)
        elif msgid == svc_Menu:                        # 29
            br.read(32)
            br.p += br.read(16)
        elif msgid == svc_GameEventList:               # 30
            br.read(16)
        elif msgid == svc_GetCvarValue:                # 31
            br.read(INVAR); br.read_string()
        elif msgid == svc_CmdKeyValues:                # 32
            br.read(INVAR); br.read(INVAR); br.read(INVAR)
        elif msgid == NETMSG_SetConVar:                # 5
            n = br.read(8)
            for _ in range(n):
                br.read_string(); br.read_string()
        elif msgid == NETMSG_StringCmd:                # 4
            br.p += br.read(16)
        elif msgid == svc_SendTable:                   # 9
            br.read(16)
        elif msgid == clc_VoiceData:                   # 10
            br.read(32); br.read(32); br.read(8)
        elif msgid == clc_BaselineAck:                 # 11
            br.read(32)
        elif msgid == clc_ListenEvents:                # 12
            br.read(32); br.read(32)
        elif msgid == clc_RespondCvarValue:            # 13
            br.read(INVAR); br.read(INVAR); br.read(INVAR); br.read(INVAR)
        elif msgid == clc_FileCRCCheck:                # 14
            br.read(INVAR); br.read(32)
        elif msgid == clc_CmdKeyValues:                # 16
            br.read(INVAR); br.read(INVAR); br.read(INVAR)
        elif msgid == clc_ClientInfo:                  # 8
            br.read(32); br.read(32); br.read(1)
            br.read(32); br.read_string(); br.read(8); br.read_string()
        elif msgid == clc_FileMD5Check:                # 17
            br.read_string()
        else:
            return False
        return br.p <= nbits
    except (IndexError, struct.error):
        return False


def _skip_packet_entities(br, nbits):
    """svc_PacketEntities needs sendtable history; we only need the bit
    cursor to land back on a message boundary, so walk the fixed header."""
    br.read(INVAR)                       # entity index
    br.read(INVAR)                       # max index
    br.read(INVAR)                       # bits
    if br.read(1) == 0:
        br.read(BITS)                    # legacy: sendtable bits
        br.read(8)                       # isDelta
        br.read(32)                      # updatebaseline
        br.read(32)                      # baseline
        br.read(32)                      # interval
        br.read(32)                      # multiplier
    return True


def load_commands(path, nframes=None, probe=False):
    """Return (header, {frame_tick: [Ucmd]}, nframes, msgid_histogram)."""
    b = open(path, 'rb').read()
    h = read_header(path)
    start = 1072 + 5 + h['signonlen']
    fr = frame_offsets(b, start, nframes)
    cmds = {}
    ids = {}
    for i, (off, tick, ln) in enumerate(fr):
        payload = b[off + FRAME_HDR:off + FRAME_HDR + ln]
        if not payload:
            continue
        moves, endp = parse_packet(payload, probe=probe, ids=ids)
        for cm in moves:
            if cm:
                cmds[tick] = cm
        if probe and i < 3:
            print('frame tick=%d len=%d msgs=%s consumed=%d bits' % (tick, ln, ids, endp))
    return h, cmds, len(fr), ids


if __name__ == '__main__':
    p = sys.argv[1]
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    h, c, nfo, ids = load_commands(p, nf, probe=True)
    print(h)
    print('frames=%d with-move=%d' % (nfo, len(c)))
    print('msgids seen:', sorted(ids.items(), key=lambda kv: -kv[1])[:10])
    for t in sorted(c)[:6]:
        for u in c[t][:3]:
            print('  tick=%-6d cmd#%-6d fwd=%-8.2f side=%-8.2f yaw=%-8.2f pitch=%-7.2f buttons=0x%x' %
                  (t, u.command_number, u.forwardmove, u.sidemove, u.y, u.p, u.buttons))
