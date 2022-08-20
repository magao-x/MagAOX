#ifndef _PLATFORM_H
#define _PLATFORM_H

#ifdef _WIN32
 #include <windows.h>
 #define WINDOWS 1
#else
 #define HANDLE int
#endif

#endif
