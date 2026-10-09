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
	{
		if (m_iChokedRun == 0)
			m_iRunStart = pCmd->command_number;
		m_iChokedRun++;
	}
}

const BufferedCmd_t* CPacketBuffer::Get(int iCommandNumber) const
{
	if (iCommandNumber <= 0)
		return nullptr;
	const auto& t = m_tCmds[iCommandNumber % MULTIPLAYER_BACKUP];
	return t.iCommandNumber == iCommandNumber ? &t : nullptr;
}

// Slice 2: server-duck the released run (fake duck). The engine transmits
// the held backlog later this frame; rewriting slots now lands before send.
// Local prediction never ducks (only the queue is touched), so locally you
// stand (see/shoot over cover) while the server sees duck. Attacking now =
// this tick untouched (stand to kill); history keeps its attack bits (kill
// from duck). Peek-cancel (neutralizing the run) is Phase 3 policy.
void CPacketBuffer::Release(CUserCmd* pCmd, bool bSendPacket)
{
	if (!pCmd || !bSendPacket || m_iChokedRun <= 0)
		return;
	if (!Vars::Packet::FakeDuck.Value)
		return;

	for (int i = m_iRunStart; i < pCmd->command_number; i++)
	{
		const int iSlot = i % MULTIPLAYER_BACKUP;
		CUserCmd* pEngine = &I::Input->m_pCommands[iSlot];
		CVerifiedUserCmd* pVer = &I::Input->m_pVerifiedCommands[iSlot];
		if (!pEngine || !pVer || pEngine->command_number != i || pVer->m_cmd.command_number != i)
			continue; // slot reused, never touch foreign ticks
		pEngine->buttons |= IN_DUCK;
		pVer->m_cmd.buttons |= IN_DUCK;
		pVer->m_crc = CRC32_ProcessSingleBuffer(&pVer->m_cmd, sizeof(CUserCmd));
	}
}
