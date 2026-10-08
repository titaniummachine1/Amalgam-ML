#pragma once
// Crash-triage boot log: append-only, capped, zero-dependency (raw Win32 file
// writes, no CRT streams, safe from DllMain-adjacent contexts). After a crash,
// the TAIL of %TEMP%\amalgam_boot.log shows exactly which stage died.
// Stages: BOOT (load) -> MENU (first menu render) -> CREATEMOVE (first ticks)
// -> BEHAV (first Predict call). Removed once the crash is found.
#include <windows.h>

inline void BootLog(const char* stage)
{
	static LONG s_n = 0;
	if (InterlockedIncrement(&s_n) > 300)
		return;
	char tmp[MAX_PATH] = {};
	if (!GetTempPathA(MAX_PATH, tmp))
		return;
	char path[MAX_PATH] = {};
	// manual concat (no CRT snprintf dependency surprises in hot paths)
	int i = 0;
	for (const char* p = tmp; *p && i < MAX_PATH - 24; p++)
		path[i++] = *p;
	const char* tail = "amalgam_boot.log";
	for (const char* p = tail; *p && i < MAX_PATH - 1; p++)
		path[i++] = *p;
	path[i] = 0;
	HANDLE h = CreateFileA(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
		nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (h == INVALID_HANDLE_VALUE)
		return;
	char line[96] = {};
	int j = 0;
	for (const char* p = stage; *p && j < 90; p++)
		line[j++] = *p;
	line[j++] = '\n';
	DWORD w = 0;
	WriteFile(h, line, (DWORD)j, &w, nullptr);
	CloseHandle(h);
}
