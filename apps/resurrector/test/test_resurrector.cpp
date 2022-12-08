/// Sample program to manage a group of hexbeater processess

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
  * Build:  g++ -I.. -I../../../INDI/INDI test_resurrector.cpp -o test_resurrector
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

#include "resurrector.hpp"

int
main(int argc, char** argv)
{
    resurrectorT<> resurr;

    // Parse command-line tokens, open FIFOs, start hexbeater processes.
    // N.B. each token should be of the form name=executable
    for (int iargc=1; iargc < argc; ++iargc)
    {
        std::string arg{argv[iargc]};

        size_t iequ = arg.find_first_of('=');
        if (iequ == std::string::npos) { continue; }
        if (iequ == (arg.size()-1)) { continue; }
        if (iequ == 0) { continue; }

        std::string driver_name(arg.substr(0,iequ));
        std::string argv0(arg.substr(iequ+1));

        // Open the FIFO for this hexbeater in the fifos/ subdirectory
        // The FIFO path will be ./fifos/<name>.hb
        int newfd = resurr.open_hexbeater(argv0, driver_name, "./fifos", NULL);

        // Start (fork) the hexbeater process
        resurr.start_hexbeater(newfd,1);
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
