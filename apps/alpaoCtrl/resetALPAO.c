/*
Compile:
gcc resetALPAO.c -o build/resetALPAO -lasdk

Call:
./resetALPAO <serialnumber>
*/

/* System Headers */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stddef.h>

/* Alpao SDK C Header */
#include "asdkWrapper.h"

/* Reset mirror values */
int resetMirror(char * serial)
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
    ret = asdkReset( dm );
    dm = NULL;

    return ret;
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

    int ret = resetMirror(serial);
    
    /* Print last error if any */
    asdkPrintLastError();

    return ret;
}