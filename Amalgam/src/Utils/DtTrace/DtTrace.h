#pragma once
// Melee/DT debug trace: append-only text log at %TEMP%\amalgam_dt.log, capped
// per session. Formatted lines (positions, bank, goal, flags) so a session can
// be replayed after unload. Debug-only: remove once DT melee is verified.
#include <windows.h>
#include <cstdio>
#include <cstdarg>

inline void DtTrace(const char* fmt, ...)
{
	static LONG s_n = 0;
	if (InterlockedIncrement(&s_n) > 60000)
		return;
	char tmp[MAX_PATH] = {};
	if (!GetTempPathA(MAX_PATH, tmp))
		return;
	char path[MAX_PATH] = {};
	snprintf(path, sizeof(path), "%samalgam_dt.log", tmp);
	char line[384] = {};
	va_list args;
	va_start(args, fmt);
	vsnprintf(line, sizeof(line), fmt, args);
	va_end(args);
	FILE* f = nullptr;
	if (fopen_s(&f, path, "a") != 0 || !f)
		return;
	fprintf(f, "%s\n", line);
	fclose(f);
}
