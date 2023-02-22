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
static bool verbose{false};

/// Signal handler:  exit on any signal caught
void
sigusr2_handler(int sig, siginfo_t *si, void *unused)
{
    if (verbose) { std::cerr << "Received SIGUSR2" << std::endl; }
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
    extern void stdout_stderr_redirect(std::string);
    resurrectorT<> resurr(&stdout_stderr_redirect);

    setup_SIGUSR2_handler();

    // Get MagAOX role and build process list file pathname e.g.
    // export MAGAOX_ROLE=vm; path=/opt/MagAOX/config/proclist_vm.txt
    std::string proclist_role = get_magaox_proclist_role(argc, argv);

    if (get_verbose_arg(argc, argv)) { resurr.set_resurr_logging(); }
    else { resurr.clr_resurr_logging(); }

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
        // Rewind proclist file, then parse proclist file again for each
        // HBM:  open FIFO; start resurrectee (HexBeater) process
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

            // Check for a FIFO fd with same driver name is already in
            // the list; find_hbm_by_name() will return -1 if there is
            // no matching driver name the list
            int newfd = resurr.find_hbm_by_name(driver_name);
            bool logged{false};
            logged = false;

            // N.B. if this action is the result of a SIGUSR2, and any
            //      item read from the proclist file has kept the same
            //      driver name but has had its argv0 updated, then the
            //      pending_close_all_close() call above should have
            //      stopped that item and cleared it from the hbm list,
            //      in which case it should not have been found by the
            //      find_hbm_by_name() above.

            // N.B. the start_hexbeater() call below will **NOT** fork a
            //      new process if it finds a running process with the
            //      same argv0 and driver name in the /proc/ filesystem.
            //
            //      So if this action is the result of a SIGUSR2, and
            //      any such processes were hung, but not yet expired,
            //      should have been stopped manually by issuing a
            //      "kill -USR2 {pid}" command

            if (newfd<0) {
                // If a FIFO with the driver name is not in the list,
                // then open a FIFO for this hexbeater in the FIFOs
                // directory; FIFO path will be /.../fifos/<name>.hb
                // N.B. open_hexbeater() would return -1 if the driver
                //      name was already in the list, but that will not
                //      happen here because of the find_hbm_by_name()
                //      call above
                newfd = resurr.open_hexbeater
                            (argv0, driver_name, IRMAGAOX_fifos, NULL);

                if (newfd<0) {
                    // If neither an existing FIFO is found nor a new
                    // FIFO is opened, then resurrector logs the erro
                    // and does nothing
                    perror(("Failed to open Hexbeater FIFO["
                           + driver_name +"," + argv0 +"]").c_str()
                          );
                    continue;
                }

                // Write the new HBM's info to STDERR
                resurr.fd_to_stream(std::cerr, newfd);
                logged = true;
            }

            // The first INDI server is started immediately; all other
            // processes (INDI drivers, non-indiservers) are delayed
            if (driver_name.substr(0,2)!="is")
            {
                if (logged)
                {
                    std::cerr << " [delayed start]" << std::endl;;
                }
                // Append FD to list and move on to next proclist line
                fd_indidrivers.push_back(newfd);
                continue;
            }

            // To here, this is INDI server ("is" prefix on driver name)

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

        // Delay if there are drivers to start and the INDI server was
        // just now started above
        if (fd_indidrivers.size() && isfd > -1)
        {
            std::cerr << "Delay 5s to start INDI drivers" << std::endl;
            timeval tv = {5,0};
            select(1,0,0,0,&tv);
        }

        // Start any selected INDI drivers
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

        if (verbose) { std::cerr << "Acting on SIGUSR2" << std::endl; }

        no_SIGUSR2_yet = true;

    } while (true);

    return 0;
}
