#pragma once
#include "../../../SDK/SDK.h"

// PacketBuffer, slice 1: capture only. Every CreateMove tick snapshots the
// final post-features cmd into a ring indexed exactly like the engine's own
// command backup (command_number % MULTIPLAYER_BACKUP), plus whether the
// tick was choked. No engine writes, no behavior change. Later slices:
// rewrite the engine queue from edited buffer (modifiers), peek rewind,
// warp-inside-choke, cancel refund.
struct BufferedCmd_t
{
	int iCommandNumber = 0;
	int iTickCount = 0;
	Vec3 vViewAngles = {};
	float flForwardMove = 0.f;
	float flSideMove = 0.f;
	float flUpMove = 0.f;
	int iButtons = 0;
	bool m_bChoked = false;
};

class CPacketBuffer
{
private:
	BufferedCmd_t m_tCmds[MULTIPLAYER_BACKUP] = {};
	int m_iChokedRun = 0;
	int m_iRunStart = 0;

	// Neutralize a withheld run in the engine queue (pivot rewrite).
	// Slot-guarded: never touches reused slots.
	void DropChokedRun(int iRunStart, int iCommandNumber);

public:
	// Fake-peek state (published by CMisc::AutoPeek pre-phase each tick;
	// rest driven here at CreateMove end).
	bool m_bPeekPlaced = false;
	Vec3 m_vPeekPivot = {};
	Vec3 m_vPeekPivotAng = {}; // viewangles at placement: abort rewrites history to these
	bool m_bPeekCommitted = false;
	bool m_bPeekReturning = false;
	bool m_bPeekRun = false;
	int m_iPeekTicks = 0;

	// Withhold predicate for FakeLag: out peeking, no shot yet, no abort.
	// Lets fakelag own the choke (single choke owner); RunPeek keeps
	// commit/abort/warp-back and never forces choke itself.
	bool PeekWantsChoke(CTFPlayer* pLocal) const;

	void Snapshot(CUserCmd* pCmd, bool bSendPacket);
	// Fake peek driver (System A): withhold/abort/commit/warp-back. Runs at
	// CreateMove end (post-aimbot), may clear *pSendPacket and request dumps.
	void RunPeek(CTFPlayer* pLocal, CUserCmd* pCmd, bool* pSendPacket);
	// Slice 2: rewrite the engine's queued cmds for the released run (see
	// .cpp). Called at CreateMove end on release ticks; the engine transmits
	// later this frame.
	void Release(CUserCmd* pCmd, bool bSendPacket);
	const BufferedCmd_t* Get(int iCommandNumber) const;
};

// Packet-domain options live with the packet module (NOT Vars.h): editing
// here recompiles ~4 files instead of the world.
namespace Vars {
	NAMESPACE_BEGIN(Packet)
		CVar(FakeDuck, "Fake duck", false);
	NAMESPACE_END(Packet)
}

ADD_FEATURE(CPacketBuffer, PacketBuffer);
