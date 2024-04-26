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
#include <cstring>
#include <iomanip>
#include <iostream>

#include "resurrector_indi.hpp"

static bool no_SIGUSR1_yet{true};
static bool no_SIGUSR2_yet{true};
static bool logging{false};
static bool verbose{false};

/// Signal handler:  exit on any signal caught
void
sigusr12_handler(int sig, siginfo_t *si, void *unused)
{
    static_cast<void>(si);
    static_cast<void>(unused);
    
    if (logging)
    {
        std::cerr << "Received signal[" 
                  << strerror(sig)
                  << "]"
                  << std::endl;
    }
    no_SIGUSR1_yet = sig==SIGUSR1 ? false : no_SIGUSR1_yet;
    no_SIGUSR2_yet = sig==SIGUSR2 ? false : no_SIGUSR2_yet;
}

/// Ignore some signals, establish handlers for others
void setup_SIGUSR12_handler(int iSIGUSRn)
{
    if (iSIGUSRn != SIGUSR1 && iSIGUSRn != SIGUSR2)
    {
        std::cerr
        << "resurrector_indi@setup_SIGUSR12_handler:  "
        << "Unknown signal argument["
        << iSIGUSRn << "(" << strerror(iSIGUSRn) << ")]"
        << "; exiting ..."
        << std::endl;
        exit(1);
    }

    int istat = -1;
    struct sigaction sa = {};

    // Catch SIGUSR1 or SIGUSR2 in sigusr12_handler(...) above
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = sigusr12_handler;
    errno = 0;
    istat = sigaction(iSIGUSRn, &sa, 0);
    if (istat < 0 || logging)
    {
        std::cerr
        << "resurrector_indi:  "
        << "sigaction(" << strsignal(iSIGUSRn) << ")=" << istat
        << "; errno=" << errno << "[" << strerror(errno) << "]"
        << std::endl;
        perror(iSIGUSRn==SIGUSR1 ? "sigaction/USR1" : "sigaction/USR2");
        if (istat) { exit(1); }
    }
}

int
main(int argc, char** argv)
{
    extern void stdout_stderr_redirect(std::string);
    bool nor{get_no_output_redirect_arg(argc,argv)};
    logging = get_logging_arg(argc, argv);
    verbose = get_verbose_arg(argc, argv);
    resurrectorT<> resurr(nor ? nullptr : &stdout_stderr_redirect);

    if (logging) { resurr.set_resurr_logging(); }
    else         { resurr.clr_resurr_logging(); }

    if (verbose) { resurr.set_resurr_verbose_logging(); }
    else         { resurr.clr_resurr_verbose_logging(); }

    // Get MagAOX role and build process list file pathname e.g.
    // export MAGAOX_ROLE=vm; path=/opt/MagAOX/config/proclist_vm.txt
    std::string proclist_role = get_magaox_proclist_role(argc, argv);

    setup_SIGUSR12_handler(SIGUSR1);
    setup_SIGUSR12_handler(SIGUSR2);

    do
    {
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

        std::string dpfx;  // Driver prefix:  "-" or <anything>:
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
        while (EOF != (rnp=read_next_process(f,driver_name,exec,dpfx)))
        {
            if (2 != rnp) { continue; }
            argv0 = IRMAGAOX_bin + std::string("/") + exec;
            if (dpfx!="-" && dpfx!="py:" && dpfx != "nhb:")
            {
                driver_name = dpfx + driver_name;
            }
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

        bool sshDiggers{false};

        while (EOF != (rnp=read_next_process(f,driver_name,exec,dpfx)))
        {
            if (2 != rnp) { continue; }

            argv0 = IRMAGAOX_bin + std::string("/") + exec;
            if (dpfx!="-" && dpfx!="py:" && dpfx != "nhb:")
            {
                driver_name = dpfx + driver_name;
            }

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

            // N.B. the start_hexbeater() calls below will **NOT** fork
            //      a new process if it finds a running process with the
            //      same argv0 and driver name in the /proc/ filesystem.
            //
            //      So if this action is the result of a SIGUSR2, and
            //      any such processes were hung, but not yet expired,
            //      should have been stopped manually by issuing a
            //      "kill -USR2 {pid}" command

            if (newfd<0)
            {

                // Default initial delay is 10s; use prefix to implement
                // special cases for maximum intial delay i.e. for
                // Hexbeaters that do not send hexbeats, so resurrector
                // will never detect their Hexbeats as having expired
                int init_delay = 10;
                if      (dpfx=="-")    { init_delay = -1; }
                else if (dpfx=="nhb:") { init_delay = -1; }
                else if (dpfx=="py:")  { init_delay = -1; }
                else                   { init_delay = 10; }

                // If a FIFO with the driver name is not in the list,
                // then open a FIFO for this hexbeater in the FIFOs
                // directory; FIFO path will be /.../fifos/<name>.hb
                // N.B. open_hexbeater() would return -1 if the driver
                //      name was already in the list, but that will not
                //      happen here because of the find_hbm_by_name()
                //      call above
                newfd = resurr.open_hexbeater(init_delay
                           , argv0, driver_name, IRMAGAOX_fifos.c_str()
                           , NULL);

                if (newfd<0)
                {
                    // If neither an existing FIFO is found nor a new
                    // FIFO is opened, then resurrector logs the error
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

            // Speical case for sshDigger (non-INDI) drivers:
            // - start them now, so network ports will be available 
            if (exec=="sshDigger")
            {
                if (logged)
                {
                    std::cerr << " [immmediate start]" << std::endl;
                }
                resurr.start_hexbeater(newfd,10);
                sshDiggers = true;
                continue;
            }

            // The first INDI server will drop through the next two if
            // clauses and be started immediately;
            //
            // All other processes (INDI drivers, non-indiservers) are
            // delayed by being
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

            // Save the FD of the first x/indiserver hexbeater process
            isfd = newfd;
        }

        fclose(f);

        if (isfd > -1)
        {
            if (sshDiggers)
            {

                // Delay if there are were sshDiggers started
                if (fd_indidrivers.size())
                {
                    std::cerr << "Delay 5s after starting sshDigger"
                              << std::endl;
                    timeval tv = {5,0};
                    select(1,0,0,0,&tv);
                }
            }
            // If an x/indiserver should be started (forked), do it now
            // N.B. 10 is the restart limit
            resurr.start_hexbeater(isfd,10);

            // Delay if there are also drivers to start
            if (fd_indidrivers.size())
            {
                std::cerr << "Delay 5s after starting INDI server"
                          << std::endl;
                timeval tv = {5,0};
                select(1,0,0,0,&tv);
            }
        }

        // Start any selected INDI drivers
        for ( auto fd : fd_indidrivers)
        {
            resurr.start_hexbeater(fd,10);
        }

        // Run the select/read/check/restart cycle
        // Refer to resurrector.hpp and HexbeatMonitor.hpp for details
        // Exit loop when either SIGUSR1 or SIGUSR2 signal received,
        // EITHER to re-read the proclist for SIGUSR2,
        // OR to stop all children for sIGUSR1 and then exit
        do
        {
            struct timeval tv{1,0};
            resurr.srcr_cycle(tv);
        } while (no_SIGUSR2_yet && no_SIGUSR1_yet);

        if (logging) {
          std::cerr << "Acting on SIGUSR"
                    << (no_SIGUSR2_yet ? "1 (exit)"
                                       : "2 (re-read proclist)"
                       )
                    << std::endl;
        }

        // Ensure no_SIGUSR2_yet is reset
        no_SIGUSR2_yet = true;

    } while (no_SIGUSR1_yet);

    if (!no_SIGUSR1_yet)
    {
        // Set all HBMs to be closed, then close them
        resurr.pending_close_all_set(true);
        resurr.pending_close_all_close();
    }

    return 0;
}
