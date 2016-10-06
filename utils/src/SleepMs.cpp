/*
Copyright 2016 Tommi M. Tykk�l�

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <SleepMs.h>

#if defined(_WIN32) || defined(_WIN64)
	#include <Windows.h>
#else
	#include <unistd.h>
#endif

void sleepMs(double delayMs) {
#if defined(_WIN32) || defined(_WIN64)
		Sleep(DWORD(delayMs));
#else
		usleep(delayMs*1000.0);
#endif
}


