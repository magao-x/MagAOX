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
const char CGraphFSMProtoHardwareMmapper::FpgaBusEmulationPathName[] = "/home/summer/.UACGraph/PZTFpgaBusEmulator.ram";

const std::string s_devmem{"/dev/mem"};

int CGraphFSMProtoHardwareMmapper::open(std::string& the_device
                                       , int& FpgaHandle
                                       , CGraphFSMHardwareInterface*& FpgaBus
                                       )
{
    if (FpgaHandle > 0) { return(EBADF); } //already open?
    if (NULL != FpgaBus) { return(EALREADY); } //already mapped?

    bool is_devmem = (s_devmem == the_device);
    if (is_devmem)
    {
        FpgaHandle = ::open("/dev/mem", O_RDWR | O_SYNC);
    }
    else
    {

        FpgaHandle = ::open(the_device.c_str()
                           , O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP
                           ); //O_NONBLOCK?
    }

    if (FpgaHandle <= 0)
    {
        fprintf(stderr, "\nMapFpgaBus(): error in open(/dev/mem): %ld.\n", (long int)errno);
        return errno;
    }
    else
    {
        if (!is_devmem)
        {
            int ft = ftruncate(FpgaHandle, sizeof(CGraphFSMHardwareInterface));  //Make sure the file is big enough for all read/writes to mmap
            if (ft < 0)
            {
                    perror("\nMapFpgaBus(): error in ftruncate()\n");
                    return errno;
            }
        }

        // Using the off_t consts seems suspect (like ~ is just gonna turn all the high bits on?)
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
            fprintf(stderr, "\nMapFpgaBus(): error in mmap(): %ld.\n", (long int)errno);
            return errno;
        }
        else
        {
            fprintf(stderr, "\nMapFpgaBus(): %s Mapped at: %p.\n", the_device.c_str(), FpgaBus);
        }
    }

    return(0);
}

int CGraphFSMProtoHardwareMmapper::close(int& FpgaHandle, CGraphFSMHardwareInterface*& FpgaBus)
{
    if (NULL != FpgaBus)
    {
        int unmap = munmap(FpgaBus, sizeof(CGraphFSMHardwareInterface));
        if (unmap < 0)
        {
            perror("\nCGraphFSMProtoHardwareMmapper::close(): error in munmap() ");
                return errno;
        }
        FpgaBus = NULL;
    }
    else { return(EALREADY); }

    if (FpgaHandle > 0)
    {
        int closed = ::close(FpgaHandle);
        if (closed < 0)
        {
            perror("\nCGraphFSMProtoHardwareMmapper::close(): error in close() ");
            return errno;
        }
        FpgaHandle = 0;
    }
    else { return(EBADF); }

    return(0);
}

int CGraphFSMProtoHardwareMmapper::read(const CGraphFSMHardwareInterface* FpgaBus, const size_t Address, void* Buffer, const size_t Len)
{
    if ( (NULL == FpgaBus) || (NULL == Buffer) ) { return(EINVAL); }

    if ( (Address + Len) > sizeof(CGraphFSMHardwareInterface) ) { return(EFAULT); }

    //Read data from fpga:
    memcpy(Buffer, ((uint8_t*)FpgaBus) + Address, Len);

    return(0);
}

int CGraphFSMProtoHardwareMmapper::write(CGraphFSMHardwareInterface* FpgaBus, const size_t Address, const void* Buffer, const size_t Len)
{
    if ( (NULL == FpgaBus) || (NULL == Buffer) ) { return(EINVAL); }

    if ( (Address + Len) > sizeof(CGraphFSMHardwareInterface) ) { return(EFAULT); }

    //Write data to fpga:
    memcpy(((uint8_t*)FpgaBus) + Address, Buffer, Len);

    return(0);
}

//EOF
