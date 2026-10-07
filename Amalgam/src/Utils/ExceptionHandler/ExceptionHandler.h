#pragma once
#include "../Macros/Macros.h"
#include <Windows.h>

class CExceptionHandler
{
public:
	void Initialize(LPVOID lpParam = nullptr);
	void Unload();

	// Cooperative silence window: while held on this thread, the filter
	// stays out of the way (no log, no dialog) so guarded code can recover
	// via SEH. Other threads are unaffected.
	void PushSilent();
	void PopSilent();
	bool IsSilent();
};

ADD_FEATURE_CUSTOM(CExceptionHandler, ExceptionHandler, U);