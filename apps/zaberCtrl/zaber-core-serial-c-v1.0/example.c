/*
 * This file is intended to give examples of typical operations one might
 * want to perform using the Zaber Thin API in C. The examples may need to
 * be adapted slightly to work with your program.
 *
 * Here the example assumes a single device is connected to "/dev/ttyUSB0",
 * using the ASCII protocol at the default baud rate (115200). It also assumes
 * it is being compiled and run on a machine using glibc, eg. Linux.
 */

/* this #define is required for nanosleep: see feature_test_macros(7) */
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdio.h>
#include <string.h>

#include "za_serial.h"

void poll_until_idle(z_port port)
{
    char reply[256] = { 0 };
	struct za_reply decoded_reply;
    const struct timespec ts = { 0, 100000000 }; /* 100mil nanosec = 100ms */
    
	/* We use za_decode() to decode this string into more manageable parts,
	 * sorting them into the fields of a za_reply struct, then we test
	 * the device_status field. Possible values for device_status are "IDLE"
	 * and "BUSY". */
    for(;;)
    {
        za_send(port, "/\n");
        za_receive(port, reply, sizeof(reply));
		za_decode(&decoded_reply, reply);

        if(strncmp(decoded_reply.device_status, "BUSY", 4) == 0)
		{
            nanosleep(&ts, NULL); /* If we're busy, wait then try again */
		}
        else 
		{
			break;
		}
    }
}

/* We take some shortcuts here in main() for the sake of demonstration and code
 * clarity, namely not checking every function's return value. Special care has
 * been taken in this API to provide meaningful return values, and we encourage
 * you to check every one of them in your final production-quality code! */
int main()
{
    z_port port;
	char reply[256] = { 0 };
	char *device_name = "/dev/ttyUSB0";
	
	if (za_connect(&port, device_name) != Z_SUCCESS)
	{
		printf("Could not connect to device %s.\n", device_name);
		return -1;
	}

    za_send(port, "/home\n");
    za_receive(port, reply, sizeof(reply));
	if (reply[0] == '\0')
	{
		printf("Read no reply from device %s. "
				"It may be running at a different baud rate "
				"and/or in the binary protocol.\n", device_name);
		return -1;
	}

    poll_until_idle(port);

	/* NB: The position 100,000 microsteps may be beyond the range of travel
	 * of your device. This example was written for devices with a long
	 * travel range, such as the A-LSQ450D. Adjust this number as necessary. */
    za_send(port, "/move abs 100000\n");
    za_receive(port, reply, sizeof(reply));
    
    poll_until_idle(port);
    
    za_disconnect(port);

    return 0;
}

