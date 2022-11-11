#include "ArcusPerformaxDriver.h"

#ifdef DEBUGARCUS
#include <stdio.h>
#endif // DEBUGARCUS

//
// I'm not thrilled with the use of these globals, but it seems required to support
// the existing fnPerformax function prototypes...
//

struct libusb_context * usb_context         = 0;
AR_DWORD                libusb_ReadTimeout  = 0;
AR_DWORD                libusb_WriteTimeout = 0;

#define AR_FALSE 0
#define AR_TRUE 1

// Internal function to allow us to keep a single list of
// vendor/product codes that are compatible with this driver
int _is_performax_device_by_vendor_product(int vendor, int product) {
    if ((vendor == 0x1589) && (product == 0xa101) ) {
        return AR_TRUE;
    }
    return AR_FALSE;
}

// Internal function to figure out if the libusb_device_descriptor is a performax device
int _is_performax_device(struct libusb_device_descriptor *descriptor) {
    return _is_performax_device_by_vendor_product(descriptor->idVendor, descriptor->idProduct);
}

// Iterate through the list of usb devices present and count the number of 
// devices that we could use (based on returning true to _is_performax_device)

AR_BOOL fnPerformaxComGetNumDevices(AR_DWORD *numDevices) {
    ssize_t device_count, i;
    libusb_device **list;
    struct libusb_device_descriptor descriptor;

    if (!InitializePerformaxLibrary()) {
        return AR_FALSE;
    }

    *numDevices = 0;

    device_count = libusb_get_device_list(usb_context, &list);


    for (i = 0; i < device_count; i++) {
    
        if (0 == libusb_get_device_descriptor(list[i], &descriptor)) {
            if ( _is_performax_device(&descriptor) ) {
                (*numDevices) ++ ;
            }
        }
    }

    libusb_free_device_list(list, 1); // Unreference and remove items from the list
    return AR_TRUE;
}

// Internal function
// Return a libusb_device_descriptor for the device given by the dwNumDevice offset.
// On success, the offset number in the libusb_device list is returned.
// On error, -1 is returned.
int _get_libusb_device_offset_from_arcos_offset(libusb_device **list, ssize_t list_count, 
	 struct libusb_device_descriptor *descriptor, AR_DWORD dwNumDevice) {
int i;

    for (i = 0; i < list_count; i++) {
        // iterate through each device and find a perfmax device: 
        if (0 == libusb_get_device_descriptor(list[i], descriptor)) {
            if ( _is_performax_device(descriptor) ) {
                if (dwNumDevice) {
                    // skip to the perfmax device offset in dwNumDevice
                    dwNumDevice--;
                } else {  // This is the one we're interested in...
                    return i;
                }
            }
        }
    }

    return -1; // Couldn't find the offset requested...
}

AR_BOOL fnPerformaxComGetProductString(AR_DWORD dwNumDevice, AR_VOID *lpDeviceString, AR_DWORD dwOptions) {
    ssize_t device_count, i;
    libusb_device **list;
    struct libusb_device_descriptor descriptor;
    AR_BOOL result;
    libusb_device_handle *device_handle;

    if (!InitializePerformaxLibrary()) {
        return AR_FALSE;
    }

    result = AR_TRUE; // no error yet...

    // Get the list of all devices:
    device_count = libusb_get_device_list(usb_context, &list);

    i = _get_libusb_device_offset_from_arcos_offset(list, device_count, &descriptor, dwNumDevice);
    if (i<0) {
       result = AR_FALSE; // invlaid dwNumDevice id; we should have found the offset above...
    } else {
       if (libusb_open(list[i], &device_handle) != 0 ) {
           result = AR_FALSE; // libusb_open error
       } else {
           if (dwOptions == PERFORMAX_RETURN_SERIAL_NUMBER) {
               if (0 > libusb_get_string_descriptor_ascii(
                       device_handle, descriptor.iSerialNumber, lpDeviceString, PERFORMAX_MAX_DEVICE_STRLEN)) {
                   result = AR_FALSE; // invalid descriptor
               }
           } else if (dwOptions == PERFORMAX_RETURN_DESCRIPTION) {
               if (0 > libusb_get_string_descriptor_ascii(
                       device_handle, descriptor.iProduct, lpDeviceString, PERFORMAX_MAX_DEVICE_STRLEN)) {
                   result = AR_FALSE; // invalid descriptor
               }
           } else {
               result = AR_FALSE; // invlaid dwOption
           }
           libusb_close(device_handle);
       }
    }

    libusb_free_device_list(list, 1); // Unreference and remove items from the list
    return result;
}

int _send_urb_control(AR_HANDLE device_handle, int id) {

    return libusb_control_transfer(device_handle,
	0x40, // bmRequestType
	0x02, // bRequest,
	id,   // wValue,
	0x00, // wIndex,
	NULL, // data,
	0,    // wLength,
	libusb_WriteTimeout);
}

AR_BOOL fnPerformaxComOpen(AR_DWORD dwDeviceNum, AR_HANDLE *device_handle) {
    ssize_t device_count, i;
    libusb_device **list;
    struct libusb_device_descriptor descriptor;
    AR_BOOL result;

    if (!InitializePerformaxLibrary()) {
        return AR_FALSE;
    }

    result = AR_TRUE; // no error yet...

    // Get the list of all devices:
    device_count = libusb_get_device_list(usb_context, &list);

    i = _get_libusb_device_offset_from_arcos_offset(list, device_count, &descriptor, dwDeviceNum);
    if (i<0) {
       result = AR_FALSE; // invlaid dwNumDevice id; we should have found the offset above...
    } else {
       if (0 != libusb_open(list[i], device_handle) ) {
           result = AR_FALSE; // libusb_open error
       } 
       if (0 != libusb_claim_interface(*device_handle, 0) ) {
           result = AR_FALSE; // libusb_open error
       }
    }

    libusb_free_device_list(list, 1); // Unreference and remove items from the list

    _send_urb_control(*device_handle, 0x02); // Should document this better; it's some open command
    return result;
}

AR_BOOL fnPerformaxComClose(AR_HANDLE device_handle) {
    AR_BOOL result;

    if (!InitializePerformaxLibrary()) {
        return AR_FALSE;
    }

    _send_urb_control(device_handle, 0x04); // Should document this better; it's some close command

    result = AR_TRUE; // no error yet...

    if (0 != libusb_release_interface(device_handle, 0) ) {
        result = AR_FALSE; // libusb_open error
    }
    libusb_close(device_handle);
    return result;
}

AR_BOOL fnPerformaxComSetTimeouts(AR_DWORD dwReadTimeout, AR_DWORD dwWriteTimeout) {
    if (!InitializePerformaxLibrary()) {
        return AR_FALSE;
    }

    libusb_ReadTimeout  = dwReadTimeout;
    libusb_WriteTimeout = dwWriteTimeout;

    return AR_TRUE; // TODO: errors for wacky times?
}

AR_BOOL fnPerformaxComSendRecv(AR_HANDLE device_handle, AR_VOID *wBuffer, AR_DWORD dwNumBytesToWrite, AR_DWORD dwNumBytesToRead, AR_VOID *rBuffer) {
    if (!InitializePerformaxLibrary()) {
        return AR_FALSE;
    }

int transferred;
int result;
char buffer[4096];

    // clear any outstanding reads:
    result = libusb_bulk_transfer(device_handle, 0x82, buffer, sizeof(buffer), &transferred, libusb_ReadTimeout);
    // if the above fails, it's probably ok.  We probably don't care.

    result = libusb_bulk_transfer(device_handle, 0x02, wBuffer, dwNumBytesToWrite, &transferred, libusb_WriteTimeout);

#ifdef DEBUGARCUS
    printf("sent: %d (%d) result %d '%s'\n", transferred, (int)dwNumBytesToWrite, result, (char *)wBuffer);
#endif // DEBUGARCUS

    if (0 != result) {
        return AR_FALSE;
    }

    result = libusb_bulk_transfer(device_handle, 0x82, rBuffer, dwNumBytesToRead, &transferred, libusb_ReadTimeout);

#ifdef DEBUGARCUS
    printf("received: %d (%d) result %d - %s\n", transferred, (int)dwNumBytesToRead, result, (char *)rBuffer);
#endif // DEBUGARCUS
    if (0 != result) {
        return AR_FALSE;
    }

    return AR_TRUE;
}

AR_BOOL fnPerformaxComFlush(AR_HANDLE device_handle) {
    if (!InitializePerformaxLibrary()) {
        return AR_FALSE;
    }

    if (_send_urb_control(device_handle, 0x01) == 0) { // Should document this better; it's some flush command
        return AR_TRUE;
    }
    return AR_FALSE;
}

//the following does _not_ need to be called before using the other functions. It is safe to ignore its existence

// Since the original interface contained the above line, all functions call the initialization function
// below.  Since we have no context being passed inthe fnPerformax functions, we don't get the opportunity
// anywhere to properly close the libusb library with libusb_exit...

AR_BOOL InitializePerformaxLibrary(AR_VOID) {
    if (usb_context) {
            return AR_TRUE;
    }

    if (!libusb_init(&usb_context)) {
        return AR_TRUE;
    }

    return AR_FALSE;
}

