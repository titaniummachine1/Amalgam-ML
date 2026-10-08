#pragma once
#include "../../../SDK/SDK.h"

// CMoveLog: per-tick movement ground-truth recorder for ML calibration.
// Logs the LOCAL player's (state, usercmd) pairs - the input->trajectory data
// that SourceTV demos can never provide (remote usercmds are not recorded).
// Gated by Vars::Misc::Movement::MoveLog; zero overhead when off (one bool).
// Output: %TEMP%\amalgam_movelog_<session>.csv, one row per usercmd.
class CMoveLog
{
private:
	bool m_bOpen = false;
	int m_iLastCmd = 0;
	void OpenFile();

public:
	void CreateMove(CTFPlayer* pLocal, CUserCmd* pCmd);
};

ADD_FEATURE(CMoveLog, MoveLog);
