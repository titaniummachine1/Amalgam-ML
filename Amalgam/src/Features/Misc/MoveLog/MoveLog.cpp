#include "MoveLog.h"

#include <fstream>
#include <windows.h>

static std::ofstream g_MoveLog;
static std::ofstream g_EnemyLog;
static char g_MoveLogPath[MAX_PATH] = {};

static float CVarFloat(const char* name)
{
	static_assert(true, "");
	ConVar* v = H::ConVars.FindVar(name);
	return v ? v->GetFloat() : std::numeric_limits<float>::quiet_NaN();
}

void CMoveLog::OpenFile()
{
	char tmp[MAX_PATH] = {};
	const DWORD n = GetTempPathA(MAX_PATH, tmp);
	if (!n || n >= MAX_PATH)
		return;
	SYSTEMTIME st = {};
	GetLocalTime(&st);
	char name[MAX_PATH] = {};
	_snprintf_s(name, sizeof(name), _TRUNCATE,
		"amalgam_movelog_%04d%02d%02d_%02d%02d%02d.csv",
		st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
	char path[MAX_PATH] = {};
	_snprintf_s(path, sizeof(path), _TRUNCATE, "%s%s", tmp, name);
	g_MoveLog.open(path, std::ios::out | std::ios::trunc);
	if (!g_MoveLog.is_open())
		return;
	memcpy(g_MoveLogPath, path, sizeof(g_MoveLogPath));
	// movement cvars first (engine calibration ground truth, read via SDK)
	// This header IS the physics export: the Python rollout bakes exactly these
	// values in, so training constants always match the env the model runs in.
	g_MoveLog << "# sv_accelerate=" << CVarFloat("sv_accelerate")
		<< " sv_airaccelerate=" << CVarFloat("sv_airaccelerate")
		<< " sv_friction=" << CVarFloat("sv_friction")
		<< " sv_stopspeed=" << CVarFloat("sv_stopspeed")
		<< " sv_maxspeed=" << CVarFloat("sv_maxspeed")
		<< " sv_maxvelocity=" << CVarFloat("sv_maxvelocity")
		<< " sv_gravity=" << CVarFloat("sv_gravity")
		<< " host_timescale=" << CVarFloat("host_timescale")
		<< " phys_timescale=" << CVarFloat("phys_timescale")
		<< " cl_forwardspeed=" << CVarFloat("cl_forwardspeed")
		<< " cl_sidespeed=" << CVarFloat("cl_sidespeed")
		<< " cl_upspeed=" << CVarFloat("cl_upspeed") << "\n";
	g_MoveLog << "tick,cmd,ox,oy,oz,vx,vy,vz,eyep,eyey,fm,sm,up,buttons,mdx,mdy,maxspeed,grounded,cls,hp,life\n";
	g_MoveLog.flush();
	// enemy actuals (tick, ent, state) for offline hit-rate join against predictions
	char epath[MAX_PATH] = {};
	_snprintf_s(epath, sizeof(epath), _TRUNCATE, "%s.enemies.csv", path);
	g_EnemyLog.open(epath, std::ios::out | std::ios::trunc);
	if (g_EnemyLog.is_open())
		g_EnemyLog << "tick,ent,ox,oy,oz,eyey,grounded,cls,hp\n";
	m_bOpen = true;
}

void CMoveLog::CreateMove(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	if (!Vars::Misc::Movement::MoveLog.Value)
	{
		m_bOpen = false;
		m_iLastCmd = 0;
		return;
	}
	if (!pLocal || !pCmd || pCmd->command_number == m_iLastCmd)
		return;
	m_iLastCmd = pCmd->command_number;
	if (!m_bOpen)
	{
		OpenFile();
		if (!m_bOpen)
			return;
	}
	const Vec3 vO = pLocal->m_vecOrigin();
	const Vec3 vV = pLocal->m_vecVelocity();
	const QAngle vA = pCmd->viewangles;
	char line[512] = {};
	_snprintf_s(line, sizeof(line), _TRUNCATE,
		"%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%d,%d,%d,%.2f,%d,%d,%d,%d\n",
		pLocal->m_nTickBase(),
		pCmd->command_number,
		vO.x, vO.y, vO.z,
		vV.x, vV.y, vV.z,
		vA.x, vA.y,
		pCmd->forwardmove, pCmd->sidemove, pCmd->upmove,
		pCmd->buttons,
		(int)pCmd->mousedx, (int)pCmd->mousedy,
		pLocal->m_flMaxspeed(),
		(pLocal->m_fFlags() & 1) ? 1 : 0,
		pLocal->m_iClass(),
		pLocal->m_iHealth(),
		(int)pLocal->m_lifeState());
	g_MoveLog << line;
	// flush every 66 rows (~1s) so a crash still leaves usable data
	static int s_n = 0;
	if (++s_n % 66 == 0)
		g_MoveLog.flush();
	// enemy actuals, same tick (for offline prediction hit-rate join)
	if (g_EnemyLog.is_open())
	{
		for (auto pEntity : H::Entities.GetGroup(EntityEnum::PlayerEnemy))
		{
			if (!pEntity)
				continue;
			auto pEnemy = pEntity->As<CTFPlayer>();
			if (!pEnemy || !pEnemy->IsAlive() || pEnemy->IsDormant())
				continue;
			const Vec3 vEO = pEnemy->m_vecOrigin();
			char eline[256] = {};
			_snprintf_s(eline, sizeof(eline), _TRUNCATE, "%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d\n",
				pLocal->m_nTickBase(), pEnemy->entindex(),
				vEO.x, vEO.y, vEO.z,
				pEnemy->GetEyeAngles().y,
				(pEnemy->m_fFlags() & 1) ? 1 : 0,
				pEnemy->m_iClass(),
				pEnemy->m_iHealth());
			g_EnemyLog << eline;
		}
		if (s_n % 66 == 0)
			g_EnemyLog.flush();
	}
}
