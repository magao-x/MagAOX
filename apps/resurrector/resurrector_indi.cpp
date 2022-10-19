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

    std::string proclist_role = get_magaox_proclist_role(argc, argv);

    FILE* f = fopen(proclist_role.c_str(),"r");

    if (!f)
    {
        std::cerr << "ERROR:  failed to open MagAOX proclist role file"
                  << "[" << proclist_role
                  << "]; " << errno << "(" << strerror(errno)
                  <<  ")=errno; exiting ..."
                  << std::endl;
        return 1;
    }

    int isfd{-1};
    std::vector<int> fd_indidrivers(0);

    // Parse proclist_<role>.txt file, open FIFOs, start resurrectee
    // (hexbeater) processes.

    std::string driver_name;
    std::string exec;
    std::string argv0;
    int rnp;
    while (EOF != (rnp=read_next_process(f,driver_name,exec)))
    {
        if (2 != rnp) { continue; }

        argv0 = IRMAGAOX_bin + std::string("/") + exec;

        // Open the FIFO for this hexbeater in the FIFOs directory
        // The FIFO path will be /.../fifos/<name>.hb
        int newfd = resurr.open_hexbeater(argv0, driver_name, IRMAGAOX_fifos, NULL);

        // Write the hexbeater info to STDERR
        resurr.fd_to_stream(std::cerr, newfd);

        if (newfd<0 || driver_name.substr(0,2)!="is")
        {
            // Non-indiservers are delayed
            std::cerr << " [delayed start]" << std::endl;;
            if (newfd > -1) { fd_indidrivers.push_back(newfd); }
            continue;
        }

        // Duplicate indiservers are ignored
        if (isfd != -1)
        {
            std::cerr << " [ignored duplicate indiserver]" << std::endl;
            resurr.close_hexbeater(newfd);
            continue;
        }

        std::cerr << std::endl;

        // Start (fork) the first x/indiserver hexbeater process
        resurr.start_hexbeater(newfd,10);
        isfd = newfd;
    }

    std::cerr << "Delaying 5s to start INDI drivers" << std::endl;
    timeval tv = {5,0};
    select(1,0,0,0,&tv);

    for ( auto fd : fd_indidrivers)
    {
        resurr.start_hexbeater(fd,10);
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
