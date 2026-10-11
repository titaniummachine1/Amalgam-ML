#pragma once
#include "../../../SDK/SDK.h"

#include <string>
#include <vector>

// CDemoLoop: elementary data factory. With the toggle on, plays every .dem in
// ~/Downloads/mge_demos + orange_demos back-to-back (demo_timescale for speed)
// and logs per-tick entity states with LIVE engine values the STV extractor
// can never provide (m_vecVelocity, m_flMaxspeed, dormant flags, weapon names).
// Rows are extractor-schema-identical so the whole pipeline (verify -> DB ->
// builders -> train) consumes them unchanged; tick-drop markers go to a sidecar
// and the capture is only trusted if verification shows zero tick jumps
// (timescale must never skip ticks - that would poison supervision).
// Gated by Vars::Misc::Movement::DemoLoop; zero overhead when off.
class CDemoLoop
{
private:
	bool m_bOpen = false;
	std::vector<std::string> m_vQueue;
	size_t m_nIdx = 0;
	bool m_bDone = false;
	int m_iLastTick = -1;
	int m_iLastLoggedTick = -1;
	int m_iStall = 0;
	std::string m_sCurDemo;
	void OpenFile(const std::string& sDemo);
	void CloseFile();
	void ScanDemos();
	void StartNext();

public:
	void FrameStageNotify();
};

ADD_FEATURE(CDemoLoop, DemoLoop);
