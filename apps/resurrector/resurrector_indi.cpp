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

#include "resurrector_indi.hpp"

static bool no_SIGUSR2_yet{true};

/// Signal handler:  exit on any signal caught
void
sigusr2_handler(int sig, siginfo_t *si, void *unused)
{
    no_SIGUSR2_yet = false;
}

/// Ignore some signals, establish handlers for others
void setup_SIGUSR2_handler()
{
    int istat = -1;
    struct sigaction sa = { 0 };

    // Catch SIGUSR2 in sigusr2_handler(...) above
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = sigusr2_handler;
    errno = 0;
    istat = sigaction(SIGUSR2, &sa, 0);
    if (istat < 0) {
        std::cerr
        << "resurrector_indi:  "
        << "sigaction(" << strsignal(SIGUSR2) << ")=" << istat
        << "; errno=" << errno << "[" << strerror(errno) << "]"
        << std::endl;
        perror("sigaction/SIGUSR2");
        exit(1);
    }
}

int
main(int argc, char** argv)
{
    resurrectorT<> resurr;

    setup_SIGUSR2_handler();

    // Get MagAOX role and build process list file pathname e.g.
    // export MAGAOX_ROLE=vm; path=/opt/MagAOX/config/proclist_vm.txt
    std::string proclist_role = get_magaox_proclist_role(argc, argv);

    do {
        // Open the process list file
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

        std::string driver_name;
        std::string exec;
        std::string argv0;

        ////////////////////////////////////////////////////////////////
        // Close any HexbeatMonitors that are not in the proclist file
        ////////////////////////////////////////////////////////////////

        // => (1) Assume all HexbeatMonitors (HBMs) are to be closed
        resurr.pending_close_all_set(true);

        // => (2) Parse proclist_<role>.txt file, reverse that
        //        assumption (1 above) for each HBM in the proclist
        int rnp;
        while (EOF != (rnp=read_next_process(f,driver_name,exec)))
        {
            if (2 != rnp) { continue; }
            argv0 = IRMAGAOX_bin + std::string("/") + exec;
            resurr.pending_close_all_set_on_match(false, argv0, driver_name);
        }

        // => (3) Close any HBM that was not in the proclist
        resurr.pending_close_all_close();

        ////////////////////////////////////////////////////////////////
        // Rewind proclist file, then parse proclist file again, for
        // each HBM:  open FIFOl start resurrectee (HexBeater) process
        ////////////////////////////////////////////////////////////////

        fseek(f, 0, SEEK_SET);

        int isfd{-1};
        std::vector<int> fd_indidrivers(0);

        isfd = -1;
        fd_indidrivers.clear();

        while (EOF != (rnp=read_next_process(f,driver_name,exec)))
        {
            if (2 != rnp) { continue; }

            argv0 = IRMAGAOX_bin + std::string("/") + exec;

            // Open the FIFO for this hexbeater in the FIFOs directory
            // The FIFO path will be /.../fifos/<name>.hb
            // N.B. open_hexbeater() will return -1 if the pair
            //      [argv0,driver_name] is already in resurr
            int newfd = resurr.open_hexbeater
                            (argv0, driver_name, IRMAGAOX_fifos, NULL);

            // Skip HBMs that are already opened
            if (newfd<0) { continue; }

            // Write the HBM info to STDERR
            resurr.fd_to_stream(std::cerr, newfd);

            // INDI drivers (non-indiservers) are delayed
            if (driver_name.substr(0,2)!="is")
            {
                std::cerr << " [delayed start]" << std::endl;;
                // Append FD to list and move on to next proclist line
                fd_indidrivers.push_back(newfd);
                continue;
            }

            // To here, FD is for an INDI server ("is" prefix)

            // Duplicate indiservers are ignored
            if (isfd != -1)
            {
                std::cerr << " [ignored duplicate indiserver]"
                          << std::endl;
                resurr.close_hexbeater(newfd);
                continue;
            }

            std::cerr << std::endl;

            // Start (fork) the first x/indiserver hexbeater process
            resurr.start_hexbeater(newfd,10);
            isfd = newfd;
        }

        fclose(f);

        if (fd_indidrivers.size())
        {
            std::cerr << "Delay 5s to start INDI drivers" << std::endl;
        }
        timeval tv = {5,0};
        select(1,0,0,0,&tv);

        // Start the selected INDI drivers
        for ( auto fd : fd_indidrivers)
        {
            resurr.start_hexbeater(fd,10);
        }

        // Run the select/read/check/restart cycle
        // Refer to resurrector.hpp and HexbeatMonitor.hpp for details
        // Exit loop when SIGUSR2 signal received, to re-read proclist
        do
        {
            struct timeval tv{1,0};
            resurr.srcr_cycle(tv);
        } while (no_SIGUSR2_yet);

        no_SIGUSR2_yet = true;

    } while (true);

    return 0;
}
