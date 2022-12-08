#include "CGFSM.hpp"

int
main(int argc, char** argv)
{
    std::string sstate{"Unknown"};
    const std::string s_devmem{"/dev/mem"};
    std::string the_device{argc > 1 ? argv[1] : s_devmem};
    bool is_devmem{s_devmem == the_device};
    fprintf(stderr, "[%s]=the_device\n", the_device.c_str());
    int the_errno{0};
    std::string errmsg;
    try
    {
        int fd{-1};
        CGraphFSMHardwareInterface* pCGFSMHI{0};

        sstate = "CGFSMHI::open";
        the_errno = CGraphFSMProtoHardwareMmapper::open(the_device, fd, pCGFSMHI, errmsg);

        fprintf(stderr, "::open{%d=errno(%s);%d=fd;%p=pCGFSMHI;%lx=interface_size}\n"
                      , the_errno, strerror(the_errno), fd, pCGFSMHI
                      , ((long)(pCGFSMHI+1)) - ((long)pCGFSMHI)
                      );

        sstate += "[" + errmsg + "]";

        if (the_errno != 0 && errmsg.size()) { throw sstate; };

        if (is_devmem)
        {
            sstate = "Writing to FPGA RAM";
        }
        else
        {
            sstate = "Writing to emulated FPGA RAM";
        }

        if (pCGFSMHI
           && (!is_devmem) // do not write time to FPGA RAM
           && argc>2
           && std::string("--write-time") == std::string(argv[2])
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

        sstate = "Final CGFSMHI::close";
        the_errno = CGraphFSMProtoHardwareMmapper::close(fd, pCGFSMHI, errmsg);
        if (the_errno != 0) { throw sstate; };
        fprintf(stderr,"%s\n", errmsg.c_str());
    }
    catch (std::string s)
    {
        fprintf(stderr, "%d=errno(%s) at [%s]; error message=[%s]\n"
                      , the_errno, strerror(the_errno), s.c_str()
                      , errmsg.c_str()
                      );
        return -1;
    }
    return 0;
}
