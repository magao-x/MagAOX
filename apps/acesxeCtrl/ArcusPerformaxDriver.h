#ifndef _ARCUSPERFORMAXDRIVER_H_
#define _ARCUSPERFORMAXDRIVER_H_

//libusb standard header file
#include "libusb-1.0/libusb.h"

//these are defined by Arcus
#define PERFORMAX_RETURN_SERIAL_NUMBER	0x0
#define PERFORMAX_RETURN_DESCRIPTION	0x1
#define PERFORMAX_MAX_DEVICE_STRLEN		256

//these defines are to conveniently turn these types transparent
//note: C does not have a bool type.
typedef int				AR_BOOL;
typedef long			AR_DWORD;
typedef void			AR_VOID;
//typedef usb_dev_handle*	AR_HANDLE;
typedef libusb_device_handle*	AR_HANDLE;

//the function definitions are the same as the windows API
AR_BOOL fnPerformaxComGetNumDevices(AR_DWORD *numDevices); 
AR_BOOL fnPerformaxComGetProductString(AR_DWORD dwNumDevice, AR_VOID *lpDeviceString, AR_DWORD dwOptions);
AR_BOOL fnPerformaxComOpen(AR_DWORD dwDeviceNum, AR_HANDLE *pHandle);
AR_BOOL fnPerformaxComClose(AR_HANDLE pHandle);
AR_BOOL fnPerformaxComSetTimeouts(AR_DWORD dwReadTimeout, AR_DWORD dwWriteTimeout);
AR_BOOL fnPerformaxComSendRecv(AR_HANDLE Handle, AR_VOID *wBuffer, AR_DWORD dwNumBytesToWrite, AR_DWORD dwNumBytesToRead, AR_VOID *rBuffer);
AR_BOOL fnPerformaxComFlush(AR_HANDLE Handle);

//the following does _not_ need to be called before using the other functions. It is safe to ignore its existence
AR_BOOL InitializePerformaxLibrary(AR_VOID);

#endif

