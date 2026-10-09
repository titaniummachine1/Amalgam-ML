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
// - window/tick-cap exceeded, no shot: ABORT (rubberband). No steering —
//   the withheld tour is neutralized in Release and the client snaps to the
//   pivot the server kept seeing. m_bPeekReturning latches "hands off until
//   home" so we don't re-withhold before the snap lands.
// - shot (any attack): COMMIT (send everything incl. backlog: the shot lands
//   from the peek spot), then walk back to the pivot (warp-back dump
//   accelerates it when WarpBack is on and banked).
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
		// Immediate warp-back delivery: force-send THIS tick so the shot +
		// the whole withheld tour burst out at once (the backlog IS the warp
		// — engine sends all choked in one packet). Ticks-bank dump
		// (WarpBack below) accelerates further when banked; the full-press
		// WalkTo after it is already speed-maximized (450, engine-clamped).
		G::SendPacket = true;
	}
	else if (!m_bPeekCommitted && !m_bPeekReturning)
	{
		*pSendPacket = false;
		m_bPeekRun = true;
		if (pLocal->m_vecOrigin().DistTo2D(m_vPeekPivot) > Vars::AutoPeek::FakePeekWindow.Value || m_iChokedRun > 22)
			m_bPeekReturning = true; // abort latch: snap home, zero steering
	}
	if ((m_bPeekCommitted || m_bPeekReturning) && ++m_iPeekTicks > 66)
	{
		m_bPeekCommitted = m_bPeekReturning = m_bPeekRun = false;
		m_iPeekTicks = 0;
	}
	if (m_bPeekReturning && m_iPeekTicks == 1)
		G::SendPacket = true; // abort releases NOW (neutralized in Release): snap within ping, not window
	if (m_bPeekCommitted && Vars::AutoPeek::WarpBack.Value)
	{
		// Unconditional warp-back: goal straight to 0 (same pattern as the
		// trickstab claim — lower the goal, next Move bursts). NO bank
		// comparison: under recharge the goal rides above shifted by design,
		// so any comparison reads empty on a full 24 bank. Dumping spends the
		// shared pool; that IS the price of the ride home. WalkTo below
		// streams when truly empty.
		BootLog("PEEK warpback");
		F::Ticks.m_bWarp = true;
		F::Ticks.m_iShiftedGoal = 0;
	}
	if (m_bPeekCommitted && G::Attacking != 1)
	{
		// Shot fired: the ONLY auto-steer fake peek does — walk back to the
		// pivot (shot tick itself untouched: moves never decide hits).
		SDK::WalkTo(pCmd, pLocal, m_vPeekPivot);
		pCmd->buttons &= ~IN_JUMP;
		// Warp-home hold: bank the full-press return tour in the engine
		// backlog (never past the overchoke ceiling) and release it as a
		// burst home — re-holds after each release until home: burst-hops.
		// Needs no Ticks bank: the withheld tour IS the warp fuel.
		if (Vars::AutoPeek::WarpBack.Value && m_iChokedRun < 21)
			*pSendPacket = false;
	}
	// m_bPeekReturning (abort latch): hands off entirely. The snap handles it.
	// Transition trace (transitions only): proves withhold/abort/commit/home.
	static bool s_bTOut = false, s_bTLat = false, s_bTCom = false, s_bTPlace = true;
	const bool bTPlaced = Vars::AutoPeek::FakePeek.Value && m_bPeekPlaced;
	if (bTPlaced != s_bTPlace) { s_bTPlace = bTPlaced; BootLog(s_bTPlace ? "PEEK pivot ok" : "PEEK no pivot"); }
	const bool bOutNow = bTPlaced && pLocal->m_vecOrigin().DistTo2D(m_vPeekPivot) >= 8.f && G::Attacking != 1 && !m_bPeekCommitted && !m_bPeekReturning;
	if (bOutNow && !s_bTOut) BootLog("PEEK withhold");
	s_bTOut = bOutNow;
	if (m_bPeekReturning && !s_bTLat) BootLog("PEEK abort");
	s_bTLat = m_bPeekReturning;
	if (m_bPeekCommitted && !s_bTCom) BootLog("PEEK commit");
	s_bTCom = m_bPeekCommitted;
}

// Standalone history drop: neutralize a withheld run in the engine queue so
// the server never receives it — as if we never moved (causality overwrite,
// not rubberband). Reusable core for future packet-domain work: pass any
// [start, current) command span. Slot-guarded: never touches reused slots.
static void DropChokedRun(int iRunStart, int iCommandNumber)
{
	for (int i = iRunStart; i < iCommandNumber; i++)
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
		// Peek abort: drop the withheld tour (stand at pivot) instead of
		// transmitting it. Abort runs never contain attacks (any attack
		// commits first); movement/ducks are discarded history.
		DropChokedRun(m_iRunStart, pCmd->command_number);
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
