#include <time.h>
#include <string>
#include "CGFSM.hpp"
int main(int argc, char** argv)
{
    static_cast<void>(argc);
    static_cast<void>(argv);
    std::string sstate{"Unknown"};
    char* ge = getenv("FPGABUS_EMULATION_PATHNAME");
    fprintf(stderr, "%s=getenv(\"FPGABUS_EMULATION_PATHNAME\")\n", ge ? ge : "<no value>");
    int the_errno{0};
    try
    {
        int fd{-1};
        CGraphFSMHardwareInterface* pCGFSMHI{0};

        sstate = "CGFSMHI::open";
        the_errno = CGraphFSMProtoHardwareMmapper::open(fd, pCGFSMHI);

        fprintf(stderr, "::open{%d=errno(%s);%d=fd;%p=pCGFSMHI;%lx=interface_size}\n"
                      , the_errno, strerror(the_errno), fd, pCGFSMHI
                      , ((long)(pCGFSMHI+1)) - ((long)pCGFSMHI)
                      );
        if (the_errno != 0) { throw sstate; };

#       ifdef EMULATE_FSM_HARDWARE
        sstate = "Writing to emulated FPGA RAM";
        if (pCGFSMHI
           && argc>1
           && std::string("--write-time") == std::string(argv[1])
           )
        {
            uint32_t dac_setpoints[3];
            size_t spaddr = (size_t)&((CGraphFSMHardwareInterface*)0)->DacASetpoint;
            sstate = "Initial CGFSMHI::read of setpoints";
            the_errno = CGraphFSMProtoHardwareMmapper::read(pCGFSMHI, spaddr, dac_setpoints, 3*sizeof(uint32_t));
            if (the_errno != 0) { throw sstate; };
            fprintf(stderr,"%u,%u,%u=DAC setpoints before\n"
                          ,dac_setpoints[0], dac_setpoints[1], dac_setpoints[2]
                          );
            dac_setpoints[2] = (uint32_t)time(0);
            dac_setpoints[0] = dac_setpoints[2]++;
            dac_setpoints[1] = dac_setpoints[2]++;

            sstate = "First CGFSMHI::write of setpoints";
            the_errno = CGraphFSMProtoHardwareMmapper::write(pCGFSMHI, spaddr, dac_setpoints, 3*sizeof(uint32_t));
            if (the_errno != 0) { throw sstate; };

            sstate = "Final CGFSMHI::read check of setpoints";
            dac_setpoints[0] =
            dac_setpoints[1] =
            dac_setpoints[2] = 0;
            the_errno = CGraphFSMProtoHardwareMmapper::read(pCGFSMHI, spaddr, dac_setpoints, 3*sizeof(uint32_t));
            if (the_errno != 0) { throw sstate; };

            fprintf(stderr,"%u,%u,%u=DAC setpoints after\n"
                          ,dac_setpoints[0], dac_setpoints[1], dac_setpoints[2]
                          );
        }
#       endif//EMULATE_FSM_HARDWARE

        sstate = "Final CGFSMHI::close";
        the_errno = CGraphFSMProtoHardwareMmapper::close(fd, pCGFSMHI);
        if (the_errno != 0) { throw sstate; };
    }
    catch (std::string s)
    {
        fprintf(stderr, "%d=errno(%s) at [%s]\n"
                      , the_errno, strerror(the_errno), s.c_str());
        return -1;
    }
    return 0;
}
