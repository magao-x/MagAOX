/// \file
/// $Revision: $

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "CGraphFSMHardwareInterface.hpp"

#define FPGA_MEM_ADDR 0x08000000
#define FPGA_MAP_MASK 0x00000FFF

const uint32_t CGraphFSMHardwareInterface::DacFullScale = 0x007FFFFFUL; //2^20 - 1 (do these need to be initialized in constructor even though it's static can't remember?)
const double CGraphFSMHardwareInterface::DacDriverFullScaleOutputVoltage = 150.0; //Volts, don't get your fingers near this thing!
const double CGraphFSMHardwareInterface::PZTDriverFullScaleOutputTravel = 0.00001; //Meters; note our granularity is this / DacFullScale which is approx 10pm

const off_t CGraphFSMProtoHardwareMmapper::FpgaMmapAdress = 0x08000000UL;
const off_t CGraphFSMProtoHardwareMmapper::FpgaMmapMask = 0x00000FFFUL;

const std::string s_devmem{"/dev/mem"};

// Open either memory device (/dev/mem) or file emulating same
// Map memory or file to an address, put that address in FpgaBus
// Put any error or success message into msg
// If successful return 0, else return non-zero errno
// On input
// - FpgaHandle must be < 0 (e.g. -1)
// - FpgaBuf must be == 0
int CGraphFSMProtoHardwareMmapper::open(const std::string& the_device
                                       , int& FpgaHandle
                                       , CGraphFSMHardwareInterface*& FpgaBus
                                       , std::string& msg
                                       )
{
    char cbuf[1024];
    msg.clear();

    // Do not re-open handle or re-mmap bus address
    if (FpgaHandle > -1) { return errno=EBADF; }     // already open?
    if (NULL != FpgaBus) { return errno=EALREADY; } // already mapped?

    bool is_devmem = (s_devmem == the_device);
    if (is_devmem)
    {
        // Either open the memory device ...
        FpgaHandle = ::open("/dev/mem", O_RDWR | O_SYNC);
    }
    else
    {
        // or open the file that emulates the memory device
        FpgaHandle = ::open(the_device.c_str()
                           , O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP
                           ); //O_NONBLOCK?
    }

    if (FpgaHandle <= 0)
    {
        // Handle any error in ::open
        sprintf(cbuf, "CGraph...::open(): error in open(%s): %ld."
                    , the_device.substr(0,256).c_str(), (long int)errno
                    );
        msg = cbuf;
        return errno;
    }

    if (!is_devmem)
    {
        // The device isn't /dev/mem, so it is a file emulating /devmem.
        // => Ensure that file is big enough for any read/writes to mmap
        if (ftruncate(FpgaHandle, sizeof(CGraphFSMHardwareInterface))<0)
        {
            // Handle error in ftruncate
            sprintf(cbuf, "%s", "CGraph...::open(): error in ftruncate()");
            msg = cbuf;
            return errno;
        }
    }

    // Map the device or file of FpgaHandle to an address
    FpgaBus = (CGraphFSMHardwareInterface*)
              mmap(0
                  , sizeof(CGraphFSMHardwareInterface)
                  , PROT_READ | PROT_WRITE
                  , MAP_SHARED
                  , FpgaHandle
                  , is_devmem ? (FPGA_MEM_ADDR & ~FPGA_MAP_MASK) : 0
                  );
    if (MAP_FAILED == FpgaBus)
    {
        // Handle any error in mmap
        sprintf(cbuf, "CGraph...::open(): error in mmap(): %ld"
                    , (long int)errno
                    );
        msg = cbuf;
        return errno;
    }

    // Load message indicating success and return 0
    sprintf(cbuf, "CGraph...::open(): %s Mapped at 0x%p"
                , the_device.substr(0,256).c_str(), FpgaBus
                );
    msg = cbuf;

    return 0;
}

// Disconnect from mmap'ed device (or file emulating same)
// - Unmap memory
// - Close device
int CGraphFSMProtoHardwareMmapper::close(int& FpgaHandle
                                        , CGraphFSMHardwareInterface*& FpgaBus
                                        , std::string& msg
                                        )
{
    msg.clear();
    if (!FpgaBus) { return errno=EALREADY; }

    int unmap = munmap(FpgaBus, sizeof(CGraphFSMHardwareInterface));
    if (unmap < 0)
    {
        perror("\nCGraph...::close(): error in munmap() ");
        msg = "CGraph...::close(): error in munmap()";
        return errno;
    }

    FpgaBus = NULL;

    if (FpgaHandle < 0) { return errno=EBADF; }

    int closed = ::close(FpgaHandle);
    if (closed < 0)
    {
        perror("\nCGraph...::close(): error in close() ");
        msg = "CGraph...::close(): error in close()";
        return errno;
    }

    msg = "CGraph...::close(): successfully closed and unmapped memory";
    FpgaHandle = -1;

    return 0;
}

// Read data from fpga
int CGraphFSMProtoHardwareMmapper::read(const CGraphFSMHardwareInterface* FpgaBus, const size_t Address, void* Buffer, const size_t Len)
{
    if ( (NULL == FpgaBus) || (NULL == Buffer) ) { return errno=EINVAL; }

    if ( (Address + Len) > sizeof(CGraphFSMHardwareInterface) ) { return errno=EFAULT; }

    memcpy(Buffer, ((uint8_t*)FpgaBus) + Address, Len);

    return 0;
}

// Write data to fpga
int CGraphFSMProtoHardwareMmapper::write(CGraphFSMHardwareInterface* FpgaBus, const size_t Address, const void* Buffer, const size_t Len)
{
    if ( (NULL == FpgaBus) || (NULL == Buffer) ) { return errno=EINVAL; }

    if ( (Address + Len) > sizeof(CGraphFSMHardwareInterface) ) { return errno=EFAULT; }

    memcpy(((uint8_t*)FpgaBus) + Address, Buffer, Len);

    return 0;
}

//EOF
