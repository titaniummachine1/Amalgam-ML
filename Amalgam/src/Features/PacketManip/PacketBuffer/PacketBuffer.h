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

public:
	void Snapshot(CUserCmd* pCmd, bool bSendPacket);
	const BufferedCmd_t* Get(int iCommandNumber) const;
};

ADD_FEATURE(CPacketBuffer, PacketBuffer);
