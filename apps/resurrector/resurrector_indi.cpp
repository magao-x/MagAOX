/// Program to manage a group of INDI-related hexbeater processess

/** Uses the resurrectorT template class and HexbeatMonitor class
  *
  * A hexbeat is a hexadecimal timestamp used as a heartbeat; a
  * hexbeater is a process that generates hexbeats
  * A resurrector class instance opens FIFOs for hexbeaters to send
  * hexbeats, starts the hexbeater processes, monitors hexbeats to
  * determine the status of the managed hexbeater processes, and
  * restarts any hexbeaters which either lockup or crash.
  *
  * N.B. Refer to HexbeatMonitor.hpp for more definitions of terms
  *
  * Build:  g++ -I../../../INDI/INDI resurrector_indi.cpp -o resurrector_indi
  * Usage:  ./test_resurrector name=executable[ name=executable[ ...]]
  *
  * Each name=executable command-line argument will fork a hexbeater
  * process via the command [executable -n name] i.e. the executable is
  * the path to the executable, and name is the name of the process,
  * which will be used to locate configuration data for the hexbeater.
  * Names must be unique, but multiple hexbeaters processes may use the
  * same executable. 
  */
#include <map>
#include <random>
#include <iomanip>
#include <iostream>
#include <assert.h>

#include "resurrector_indi.hpp"

int
main(int argc, char** argv)
{
    resurrectorT<> resurr;

    FILE* f = fopen("/opt/MagAOX/config/proclist_vm.txt","r");
    std::string driver_name;
    std::string exec;
    std::string argv0;

    int isfd = -1;

    // Parse command-line tokens, open FIFOs, start hexbeater processes.
    // N.B. each token should be of the form name=executable
    int rnp;
    while (EOF != (rnp=read_next_process(f,driver_name,exec)))
    {
        if (2 != rnp) { continue; }

        argv0 = IRMAGAOX_bin + std::string("/") + exec;

        // Open the FIFO for this hexbeater in the FIFOs directory
        // The FIFO path will be /.../fifos/<name>.hb
        int newfd = resurr.open_hexbeater(argv0, driver_name, IRMAGAOX_fifos, NULL);


        // Start (fork) the hexbeater process
        resurr.fd_to_stream(std::cout, newfd);

        if (isfd==-1 && driver_name.substr(0,2)=="is")
        {
            isfd = newfd;
            std::cout << " [delayed start]";
        }

        std::cout << std::endl;

        resurr.start_hexbeater(newfd,3);
    }
    if (isfd > -1) {
        std::cout << "Delaying 5s to start indiserver" << std::endl;
        timeval tv = {5,0};
        select(1,0,0,0,&tv);
        resurr.start_hexbeater(isfd,3);
    }

    // Run the select/read/check/restart cycle
    // Refer to resurrector.hpp and HexbeatMonitor.hpp for details
    do
    {
        struct timeval tv{1,0};
        resurr.srcr_cycle(tv);
    } while (1);

    return 0;
}
