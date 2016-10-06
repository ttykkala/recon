#if defined(_WIN64)
#include <windows.h>
#else

#endif
#include <stdio.h>
#include <PageLockedMemory.h>
#include <inttypes.h>


PageLockedMemory::PageLockedMemory() {
	data = NULL;
	allocatedBytes = 0;
}

PageLockedMemory::~PageLockedMemory() {
	release();
}

bool PageLockedMemory::alloc(uint64_t requiredBytes) {
	// free if already allocated:
	if (data != NULL) { release();  }
/*#if defined(_WIN64)
	data = (unsigned char*)VirtualAlloc(NULL, requiredBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	HANDLE hProcess = GetCurrentProcess();
	SIZE_T dwMin, dwMax;
	if (!GetProcessWorkingSetSize(hProcess, &dwMin, &dwMax)) {
		printf("GetProcessWorkingSetSize failed (%d)\n", GetLastError());
		return 0;
	}
	printf("old process working set range: %llu %llu\n", dwMin, dwMax);
	if (!SetProcessWorkingSetSize(hProcess, requiredBytes, requiredBytes*2)) {
		printf("SetProcessWorkingSetSize failed (%d)\n", GetLastError());
		return 0;
	}
	if (!GetProcessWorkingSetSize(hProcess, &dwMin, &dwMax)) {
		printf("GetProcessWorkingSetSize failed (%d)\n", GetLastError());
		return 0;
	}
	printf("new process working set range: %llu %llu\n", dwMin, dwMax);
	if (!VirtualLock(data, requiredBytes)) {
		printf("memory locking failed! (%d)\n", GetLastError());
		return 0;
	}
#else*/
	data = new unsigned char[requiredBytes];
//#endif
	if (data != NULL) {
		allocatedBytes = requiredBytes;
		return true;
	}
	else return false;
}

bool PageLockedMemory::release() {
	// return if already released
	if (data == NULL) return true;
	bool success = true;
/*#if defined(_WIN64)
	success &= VirtualUnlock(data, allocatedBytes) != FALSE;
	success &= VirtualFree(data, 0, MEM_RELEASE) != FALSE;
#else*/
	delete[] data; data = NULL;
//#endif
	return success;
}

unsigned char *PageLockedMemory::ptr() {
	if (data == NULL) return NULL;
	return data;
}

unsigned char *PageLockedMemory::ptr(uint64_t byteOffset) {
	if (data == NULL) return NULL;
	return &data[byteOffset];
}
