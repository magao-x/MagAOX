#include <cstring>
#include "CGraphFSMHardwareInterface.hpp"

typedef struct CGFSMHIstr
{
    AdcAccumulator                   adcacc   ;
    AdcTimestamp                     adcts    ;
    AdcFifo                          adcfifo  ;
    CGraphFSMHardwareControlRegister CGFSMHCR ;
    CGraphFSMHardwareStatusRegister  CGFSMHSR ;
    CGraphFSMHardwareInterface       CGFSMHI  ;
} CGFSMHI;

int
main(int argc, char** argv)
{
    CGFSMHI c;
    memset(&c, 0, sizeof c);

    //c.adcacc.fprintf(stdout);
    //::fprintf(stdout,"\n");

    //c.adcfifo.fprintf(stdout);
    //::fprintf(stdout,"\n");

    //c.CGFSMHCR.fprintf(stdout);
    //::fprintf(stdout,"\n");

    //c.CGFSMHSR.fprintf(stdout);
    //::fprintf(stdout,"\n");

    //c.CGFSMHI.fprintf(stdout,&c.CGFSMHI);
    //::fprintf(stdout,"\n");

    return 0;
}
