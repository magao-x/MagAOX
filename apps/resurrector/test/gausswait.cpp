/// gausswait.cpp - program to test INDI resurrector

/** Build:
  *
  *     #PWD=.../apps/resurecttor/test/
  *     g++  -I../../../INDI/INDI gausswait.cpp -o gausswait
  *
  * Usage:
  *
  *     cd ../
  *     mkfifo fifos/aaaa
  *     mkfifo fifos/bbbb
  *     mkfifo fifos/cccc
  *     ./test_resurrector aaaa=./gausswait bbbb=./gausswait cccc=./gausswait
  *
  * This program will be forked by INDI resurrector,
  * and will perform several primary tasks:
  *
  * 1) Exit with logging to std::cerr on receipt of a SIGUSR2 signal
  * 2) Parse command-line arguments to get the process name
  * 3) Wait for a random amount of time (Note i)
  * 4) Write a hexbeat (Note ii) to a named FIFO (Note iii)
  * 5) Repeat from task (3) above, until a SIGUSR2 signal is received
  *
  * Notes
  *
  * i) the time to wait will vary, and be the absolute value from a
  *    random number with a Gaussian distribution, a mean afe 0.0s, and
  *    a standard deviation of 0.8s.
  * ii) heartbeat representing a whole second in the future
  * iii) fifos/<process_name>)
  */
#include <ctype.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>

extern "C" {
#include "strcat_varargs.c"
#include "open_named_fifo.c"
};

#include <assert.h>
#include <random>
#include <iostream>

#define TICKBITS 6
#define TICKDIVISOR (1<<TICKBITS)
#define TICKMASK (TICKDIVISOR - 1)
#define TICKUSPERTICK (1000000/TICKDIVISOR)

// Random Number Generator:  Gaussian distribution; mean=0; stddev=1.0
static std::random_device rd{};
static std::mt19937 gen{rd()};
static std::normal_distribution<> gaussian{0,1.0};

// Save some static information e.g. to be written to std::cerr on exit
const char unknown[] = { "<unknown>" };  // Process name until -n parsed
static char* myname = (char*) unknown;   // Process name after -n parsed
static struct timeval static_timeout;    // Keep track of last timeout
static int broken_pipes = 0;             // Count of successive SIGPIPEs
static int mypid = -1;                   // PID of this process

/// Signal handler:  exit on any signal caught
void
sigusr2_handler(int sig, siginfo_t *si, void *unused)
{
    static_cast<void>(si);
    static_cast<void>(unused);
    std::cerr
    << "Driver[" << myname << "]:  "
    << "PID=" << mypid
    << "; caught and exiting on ["
    << strsignal(sig)
    << "]; timeout contents={" << static_timeout.tv_sec
    << ',' << static_timeout.tv_usec
    << "}\n"
    ;
    exit(0);
}

/// Ignore some signals, establish handlers for others
void handle_SIGs()
{
    int istat = -1;
    struct sigaction sa = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    // Ignore SIGPIPE on bad write so we can handle it inline
    // Ignore SIGINT so ^C will stop parent process (resurrector) only
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sa.sa_handler = SIG_IGN;
    (void)sigaction(SIGPIPE, &sa, NULL);
    (void)sigaction(SIGINT, &sa, NULL);

    // Catch SIGUSR2 in sigusr2_handler(...) above
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = sigusr2_handler;
    errno = 0;
    for (auto isig : {SIGUSR2}) {
        istat = sigaction(isig, &sa, 0);
        if (istat < 0) {
            std::cerr
            << "Driver[" << myname << "]:  "
            << "sigaction("
            << strsignal(isig)
            << ")=" << istat
            << "; errno=" << errno
            << "[" << strerror(errno)
            << "]\n";
            perror("# sigaction/SIGPIPE");
            exit(1);
        }
    }
}

/// Generate random number, convert to timeout [struct timeval]
void
gauss_timeval(double onesigma, struct timeval *timeout)
{
    double gg = onesigma * gaussian(gen);
    int ticks = std::abs(TICKDIVISOR * gg);
    timeout->tv_usec = (TICKMASK & ticks) * TICKUSPERTICK;
    timeout->tv_sec = ticks >> TICKBITS;
    return;
}

/// Write hexbeat timestamp to FIFO
void
send_hexbeat(int fdfifo, int offset)
{
    // Generate hexbeat timestamp
    char stimestamp[18];
    sprintf(stimestamp,"%9.9lx\n",time(0)+offset);

    // Write hexbeat to FIFO
    int irtn = write(fdfifo,stimestamp,10);

    // On success, return
    if (irtn > 0)
    {
        // If previous read(s) had an error, then log recovery and reset
        if(broken_pipes)
        {
            std::cerr
            << "Driver[" << myname << "]:  "
            << "recovered\n"
            ;
            broken_pipes = 0;
        }
        return;
    }

    // Ignore successiveerrors after the third
    if (broken_pipes>2) { return; }

    // Log first three successive errors
    ++broken_pipes;
    char* pNL = strchr(stimestamp,'\n');
    if (pNL && (pNL-stimestamp)<10) { strcpy(pNL,"\\n"); }
    std::cerr
    << "Driver[" << myname << "]:  "
    << irtn
    << "=write(" << fdfifo
    << ",[" << stimestamp
    << "],10); errno=" << errno
    << "[" << strerror(errno)
    << "]\n"
    ;
    errno = 0;
}

int
main(int argc, char** argv)
{
    // Parse name, save to static memory
    assert(argc == 3);
    assert(std::string("-n") == std::string(argv[1]));
    myname = argv[2];

    // Save PID to static memory
    mypid = getpid();

    // Set up signal handling
    handle_SIGs();

    // Open hexbeat FIFO
    int fdhb = open_named_fifo(O_WRONLY, "./fifos/", myname, ".hb", NULL);
    assert(fdhb > -1);

    // Loop; exit via signal, typically SIGUSR2; N.B. SIGINT is ignored
    do
    {
        // Generate random timeout with stddev=1s, save to static memory
        struct timeval timeout;
        gauss_timeval(1.0, &timeout);
        static_timeout = timeout;

        // Send hexbeat with a timestamp 3s in the future
        send_hexbeat(fdhb, 3);

        // Use select to wait for the random timeout
        select(0, 0,0,0, &timeout);
    } while (1);                     // Repeat

    return 0;
}
