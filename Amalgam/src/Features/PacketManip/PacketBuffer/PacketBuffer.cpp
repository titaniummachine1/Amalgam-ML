#include "PacketBuffer.h"

#include "../../../Utils/BootLog/BootLog.h"
#include "../../../SDK/VarsAutoPeek.h"
#include "../../Ticks/Ticks.h"

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

// Fake peek driver (System A: causality). Decided per tick at CreateMove end:
// - home/disabled: normal, clear peek state.
// - out peeking, no shot: WITHHOLD (server keeps seeing pivot).
// - window/tick-cap exceeded, no shot: ABORT (walk back withheld, resume at
//   pivot with the tour neutralized in Release).
// - shot (any attack): COMMIT (send everything incl. backlog: the shot lands
//   from the peek spot), then warp-back dumps accelerate the base WalkTo
//   return when WarpBack is on and banked, else normal-speed return.
void CPacketBuffer::RunPeek(CTFPlayer* pLocal, CUserCmd* pCmd, bool* pSendPacket)
{
	if (!pLocal || !pCmd || !pSendPacket)
		return;
	if (!Vars::AutoPeek::FakePeek.Value || !m_bPeekPlaced)
	{
		m_bPeekCommitted = m_bPeekReturning = m_bPeekRun = false;
		m_iPeekTicks = 0;
		return;
	}
	if (pLocal->m_vecOrigin().DistTo2D(m_vPeekPivot) < 8.f)
	{
		m_bPeekCommitted = m_bPeekReturning = m_bPeekRun = false;
		m_iPeekTicks = 0;
		return;
	}
	if (G::Attacking == 1)
	{
		m_bPeekCommitted = true;
		m_bPeekReturning = false;
		m_iPeekTicks = 0;
	}
	else if (!m_bPeekCommitted)
	{
		*pSendPacket = false;
		m_bPeekRun = true;
		if (pLocal->m_vecOrigin().DistTo2D(m_vPeekPivot) > Vars::AutoPeek::FakePeekWindow.Value || m_iChokedRun > 22)
			m_bPeekReturning = true;
	}
	if ((m_bPeekCommitted || m_bPeekReturning) && ++m_iPeekTicks > 66)
	{
		m_bPeekCommitted = m_bPeekReturning = m_bPeekRun = false;
		m_iPeekTicks = 0;
	}
	if (m_bPeekCommitted && Vars::AutoPeek::WarpBack.Value && F::Ticks.GetTicks() > 0)
		F::Ticks.m_bDoubletap = true;
	else if (m_bPeekReturning)
	{
		SDK::WalkTo(pCmd, pLocal, m_vPeekPivot);
		pCmd->buttons &= ~IN_JUMP;
	}
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
	if (m_bPeekRun && !m_bPeekCommitted)
	{
		// Peek abort: neutralize the withheld tour (stand at pivot) instead
		// of transmitting it. Abort runs never contain attacks (any attack
		// commits first); movement/ducks are discarded history.
		for (int i = m_iRunStart; i < pCmd->command_number; i++)
		{
			const int iSlot = i % MULTIPLAYER_BACKUP;
			CUserCmd* pEngine = &I::Input->m_pCommands[iSlot];
			CVerifiedUserCmd* pVer = &I::Input->m_pVerifiedCommands[iSlot];
			if (!pEngine || !pVer || pEngine->command_number != i || pVer->m_cmd.command_number != i)
				continue; // slot reused, never touch foreign ticks
			pEngine->forwardmove = pEngine->sidemove = pEngine->upmove = 0.f;
			pEngine->buttons &= ~(IN_DUCK | IN_JUMP);
			pVer->m_cmd.forwardmove = pVer->m_cmd.sidemove = pVer->m_cmd.upmove = 0.f;
			pVer->m_cmd.buttons &= ~(IN_DUCK | IN_JUMP);
			pVer->m_crc = CRC32_ProcessSingleBuffer(&pVer->m_cmd, sizeof(CUserCmd));
		}
		m_bPeekRun = false;
		return;
	}
	m_bPeekRun = false;
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
