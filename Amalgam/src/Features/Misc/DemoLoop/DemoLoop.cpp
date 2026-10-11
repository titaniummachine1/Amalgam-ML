#include "DemoLoop.h"
#include "../../Simulation/BehaviorPrediction/BehaviorPrediction.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <windows.h>

// Self-contained demo-loop data factory (see header for the why).
// Schema mirrors the Rust extractor EXACTLY (ml/build_mspd.py PLAYER_COLS +
// wcls/w0/w1/w2 loadout cols) so verify/DB/builders consume captures unchanged.
// LIVE values the extractor cannot provide: m_vecVelocity (STV vel is stale),
// m_flMaxspeed (per-tick cap), dormant flags, weapon class names.

static std::ofstream g_DemoLog;
static std::ofstream g_DropLog;
static char g_DemoLogPath[MAX_PATH] = {};

void CDemoLoop::ScanDemos()
{
	m_vQueue.clear();
	char home[MAX_PATH] = {};
	if (!GetEnvironmentVariableA("USERPROFILE", home, MAX_PATH))
		return;
	const char* dirs[2] = { "\\Downloads\\mge_demos\\*.dem", "\\Downloads\\orange_demos\\*.dem" };
	for (int d = 0; d < 2; d++)
	{
		char pat[MAX_PATH] = {};
		_snprintf_s(pat, sizeof(pat), _TRUNCATE, "%s%s", home, dirs[d]);
		WIN32_FIND_DATAA fd = {};
		HANDLE h = FindFirstFileA(pat, &fd);
		if (h == INVALID_HANDLE_VALUE)
			continue;
		do
		{
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				char full[MAX_PATH] = {};
				_snprintf_s(full, sizeof(full), _TRUNCATE, "%s%s\\%s", home,
					d == 0 ? "\\Downloads\\mge_demos" : "\\Downloads\\orange_demos", fd.cFileName);
				m_vQueue.emplace_back(full);
			}
		} while (FindNextFileA(h, &fd));
		FindClose(h);
	}
	std::sort(m_vQueue.begin(), m_vQueue.end());
}

void CDemoLoop::OpenFile(const std::string& sDemo)
{
	char tmp[MAX_PATH] = {};
	if (!GetTempPathA(MAX_PATH, tmp))
		return;
	const char* base = strrchr(sDemo.c_str(), '\\');
	base = base ? base + 1 : sDemo.c_str();
	char path[MAX_PATH] = {};
	_snprintf_s(path, sizeof(path), _TRUNCATE, "%sdemoloop_%s.csv", tmp, base);
	g_DemoLog.open(path, std::ios::out | std::ios::trunc);
	if (!g_DemoLog.is_open())
		return;
	memcpy(g_DemoLogPath, path, sizeof(g_DemoLogPath));
	g_DemoLog << "tick,ent,class,ox,oy,oz,vx,vy,vz,eyep,eyey,mspd,flags,health,life,team,fresh,wcls,w0,w1,w2\n";
	char dpath[MAX_PATH] = {};
	_snprintf_s(dpath, sizeof(dpath), _TRUNCATE, "%s.drops.log", path);
	g_DropLog.open(dpath, std::ios::out | std::ios::trunc);
	m_bOpen = true;
}

void CDemoLoop::CloseFile()
{
	if (g_DemoLog.is_open())
		g_DemoLog.close();
	if (g_DropLog.is_open())
		g_DropLog.close();
	m_bOpen = false;
}

void CDemoLoop::StartNext()
{
	CloseFile();
	if (m_nIdx >= m_vQueue.size())
	{
		m_bDone = true; // queue exhausted: idle until toggle cycled
		return;
	}
	m_sCurDemo = m_vQueue[m_nIdx++];
	char cmd[MAX_PATH] = {};
	_snprintf_s(cmd, sizeof(cmd), _TRUNCATE, "playdemo \"%s\"", m_sCurDemo.c_str());
	I::EngineClient->ClientCmd_Unrestricted(cmd);
	m_iLastTick = -1;
	m_iLastLoggedTick = -1;
	m_iStall = 0;
	OpenFile(m_sCurDemo);
}

void CDemoLoop::FrameStageNotify()
{
	if (!Vars::Misc::Movement::DemoLoop.Value)
	{
		if (m_bOpen || m_nIdx)
		{
			I::EngineClient->ClientCmd_Unrestricted("stopdemo");
			CloseFile();
			m_nIdx = 0;
			m_bDone = false;
		}
		return;
	}
	if (m_vQueue.empty() && !m_bDone)
	{
		ScanDemos();
		char sc[64] = {};
		_snprintf_s(sc, sizeof(sc), _TRUNCATE, "demo_timescale %g",
			Vars::Misc::Movement::DemoLoopScale.Value);
		I::EngineClient->ClientCmd_Unrestricted(sc);
	}
	if (!I::EngineClient->IsPlayingDemo())
	{
		if (!m_bDone)
			StartNext();
		return;
	}
	const int iTick = I::EngineClient->GetDemoPlaybackTick();
	if (iTick == m_iLastTick)
	{
		if (++m_iStall > 1800)
		{
			// frozen 30s wall: corrupt demo, skip it (logged via missing range)
			I::EngineClient->ClientCmd_Unrestricted("stopdemo");
			m_iStall = 0;
		}
		return;
	}
	m_iStall = 0;
	if (iTick < m_iLastTick)
	{
		// manual playdemo / looped: rotate file so ticks stay monotonic per file
		CloseFile();
		OpenFile(m_sCurDemo);
		m_iLastLoggedTick = -1;
	}
	// tick jump = dropped ticks (timescale too aggressive or corrupt demo):
	// mark it; verification quarantines files with any markers (poison rule).
	if (m_bOpen && m_iLastLoggedTick >= 0 && iTick > m_iLastLoggedTick + 1 && g_DropLog.is_open())
		g_DropLog << "tick_jump " << m_iLastLoggedTick << " " << iTick << "\n";
	m_iLastTick = iTick;
	if (!m_bOpen || iTick == m_iLastLoggedTick)
		return;
	m_iLastLoggedTick = iTick;
	for (auto pEntity : H::Entities.GetGroup(EntityEnum::PlayerAll))
	{
		if (!pEntity)
			continue;
		auto pPlayer = pEntity->As<CTFPlayer>();
		if (!pPlayer || !pPlayer->IsAlive() || pPlayer->IsAGhost() || pPlayer->IsDormant())
			continue;
		const Vec3 vO = pPlayer->m_vecOrigin();
		const Vec3 vV = pPlayer->m_vecVelocity(); // LIVE (STV vel is stale)
		const Vec3 vA = pPlayer->GetEyeAngles();
		// live weapon classes (names match the extractor's network class strings)
		const char* wcls = "";
		const char* wld[3] = { "", "", "" };
		if (auto pWep = I::ClientEntityList->GetClientEntityFromHandle(pPlayer->m_hActiveWeapon()))
		{
			if (auto pCls = pWep->GetClientClass())
				wcls = pCls->m_pNetworkName ? pCls->m_pNetworkName : "";
		}
		for (int s = 0; s < 6; s++)
		{
			auto pW = pPlayer->GetWeaponFromSlot(s);
			if (!pW)
				continue;
			int slot = -1; float c = 0.f;
			CBehaviorPrediction::WeaponFeat(int(pW->GetClassID()), &slot, &c);
			if (slot >= 0 && slot < 3 && !wld[slot][0])
			{
				if (auto pCls = pW->GetClientClass())
					wld[slot] = pCls->m_pNetworkName ? pCls->m_pNetworkName : "";
			}
		}
		char line[1024] = {};
		_snprintf_s(line, sizeof(line), _TRUNCATE,
			"%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,1,%s,%s,%s,%s\n",
			iTick, pPlayer->entindex(), pPlayer->m_iClass(),
			vO.x, vO.y, vO.z, vV.x, vV.y, vV.z, vA.x, vA.y,
			pPlayer->m_flMaxspeed(), pPlayer->m_fFlags(),
			pPlayer->m_iHealth(), 0, pPlayer->m_iTeamNum(),
			wcls, wld[0], wld[1], wld[2]);
		g_DemoLog << line;
	}
	static int s_n = 0;
	if (++s_n % 66 == 0)
		g_DemoLog.flush();
}
