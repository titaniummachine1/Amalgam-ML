#include "PacketBuffer.h"

#include "../../../Utils/BootLog/BootLog.h"

void CPacketBuffer::Snapshot(CUserCmd* pCmd, bool bSendPacket)
{
	if (!pCmd || pCmd->command_number <= 0)
		return;
	auto& t = m_tCmds[pCmd->command_number % MULTIPLAYER_BACKUP];
	t.iCommandNumber = pCmd->command_number;
	t.iTickCount = pCmd->tick_count;
	t.vViewAngles = { pCmd->viewangles.x, pCmd->viewangles.y, pCmd->viewangles.z };
	t.flForwardMove = pCmd->forwardmove;
	t.flSideMove = pCmd->sidemove;
	t.flUpMove = pCmd->upmove;
	t.iButtons = pCmd->buttons;
	t.m_bChoked = !bSendPacket;

	// Slice-1 proof: log each choke-run release (throttled). With fakelag N
	// the releases land ~every N ticks; count PKTBUF lines to verify.
	if (bSendPacket)
	{
		if (m_iChokedRun > 0)
		{
			static int s_n = 0;
			if (s_n < 30) { s_n++; BootLog("PKTBUF release"); }
		}
		m_iChokedRun = 0;
	}
	else
		m_iChokedRun++;
}

const BufferedCmd_t* CPacketBuffer::Get(int iCommandNumber) const
{
	if (iCommandNumber <= 0)
		return nullptr;
	const auto& t = m_tCmds[iCommandNumber % MULTIPLAYER_BACKUP];
	return t.iCommandNumber == iCommandNumber ? &t : nullptr;
}
