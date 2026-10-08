#pragma once
#include "../../../SDK/SDK.h"

class CFakeLag
{
private:
	bool IsAllowed(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd);
	void Prediction(CTFPlayer* pLocal, CUserCmd* pCmd);
	void PreserveBlastJump(CTFPlayer* pLocal);
	void Unduck(CTFPlayer* pLocal, CUserCmd* pCmd);

	Vec3 m_vLastPosition = {};
	bool m_bPreservingBlast = false;
	bool m_bUnducking = false;

public:
	void Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd, bool* pSendPacket);

	int m_iGoal = 0;
	// charge reach: the fire tick must reach the server unchoked, or the
	// smack window (computed from latency+choke) mistimes. consume-once.
	bool m_bForceSend = false;
};

ADD_FEATURE(CFakeLag, FakeLag);