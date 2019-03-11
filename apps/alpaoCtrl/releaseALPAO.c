/*
Compile:
gcc releaseALPAO.c -o build/releaseALPAO -lasdk

Call:
./releaseALPAO <serialnumber>
*/

/* System Headers */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stddef.h>

/* Alpao SDK C Header */
#include "asdkWrapper.h"

/* Reset and Release */
int releaseMirror(char * serial)
{
    COMPL_STAT ret;
    asdkDM * dm = NULL;

    /* Load configuration file */
    dm = asdkInit(serial);
    if (dm == NULL)
    {
        return -1;
    }

    /* reset */
    asdkReset( dm );

    /* release connection */
    ret = asdkRelease( dm );
    dm = NULL;

    return 0;
}

/* Main program */
int main( int argc, char ** argv )
{
    char * serial;

    if (argc < 2)
    {
        printf("Serial number must be supplied.\n");
        return -1;
    }
    serial = argv[1];

    int ret = releaseMirror(serial);
    
    /* Print last error if any */
    asdkPrintLastError();

    return ret;
}