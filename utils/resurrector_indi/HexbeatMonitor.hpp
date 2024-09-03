/// Nine-digit hexadecimal newline-terminated heartbeat class
#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/types.h>

#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>

#include <dirent.h>

// /////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////
/// Convert delay to 9dnths relative to current time
/** 9dnths means "9-digit newline-terminated hexadecimal string"
  * A delay is added to the current seconds past the Epoch cf. time(2)
  * and the sum written in hexadecimal, padded with 0s to be nine
  * digits, with a terminating newline
  */
static inline std::string time_to_hb(int delay)
{
    char c20[20];
    if (delay<0)
    {
        // If delay is negative, use maximum 9dnths
        sprintf(c20,"%1.1lx\n",0xfL);
        for (int i=1; i<9; ++i) { c20[i] = *c20; }
        c20[9] = '\n';
    }
    else
    {
        // Build delay string from current time and non-negative delay
        sprintf(c20,"%9.9lx\n",time(0) + delay);
    }
    return std::string{c20};
}
// /////////////////////////////////////////////////////////////////////

// /////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////
/// HexbeaterMonitor class for monitoring a process via a heartbeat
/** A process, called a heartbeater, generates the heartbeat
  * The heartbeat is sent over some channel (FIFO) at regular intervals
  * The heartbeat (hexbeat) itself is a 9-digit newline-terminated
  * hexadecimal string (9dnths), representing a time value (Note 1)
  * after which the process is considered to have missed a (heart-)
  * beat, and either is locked up or has died or otherwise exited.
  *
  * Note 1: s past the Epoch,  cf. time(2) in man pages
  *
  * Operations
  * ==========
  *
  * Instances of this class are designed to be used within a supervisory
  * process that is monitoring multiple hexbeaters, with one instance
  * per hexbeater.
  *
  * An instance of this class is in one of three operational states:
  * 1) inactive - available to be used for a new hexbeat
  * 2) opened - file (FIFO) open for hexbeat, but no known hexbeater
  * 3) started - file (FIFO) open for hexbeat, known hexbeater process
  *
  * \todo describe transitions between states
  *
  * Public Interfaces
  * =================
  *
  *  open_hexbeater - +Open FIFO, initialize HexbeatMonitor instance
  * start_hexbeater - Find a hexbeater or start one by forking a process
  *    late_hexbeat - Check if latest value of hexbeat has expired
  *  read_hexbeater - Read data from FD, parse possible hexbeat
  *  stop_hexbeater - Stop a hexbeater process
  * close_hexbeater - Stop a running hexbeater process, close FIFO
  * find_hexbeater_pid - +Find a hexbeater PID by executable and name
  * update_restart_check_expiry - Restart logic
  *
  * - Interface description+ marked with a plus sign (+) are static
  *
  * Internals
  * =========
  *
  * Instances of this class are meant to exist in a group such as an
  * array of same.  The class itself is a model for the pieces of a
  * heartbeat-monitoring scheme:  the hexbeaters i.e. the processes
  * generating the hexbeats; the hexbeat monitor process.  The model
  * takes the viewpoint of the monitoring process.
  *
  * The m_fd and m_sel attributes compose the operational state of the
  * instance:
  *
  * m_fd (negative for inactive; non-negative for opened or started)
  *
  * - FIFO File Descriptor (FD) of FIFO (current; e.g. could be socket)
  *
  *   The File Descriptor of a file (named FIFO) that is on the
  *   receiving end of the hexbeat.  It will have a value that is either
  *   non-negative when the file (FIFO) is open, or negative (-1) if
  *   there is no open file and this instance is available to represent
  *   a new hexbeat.
  *
  * m_sel (false for inactive or opened; true for started)
  *
  * - The select(2) monitoring flag
  *
  *   Indicator that two-way communications a possible i.e. that a
  *   monitored hexbeater process is known to exist.  m_sel will be
  *   false when m_fd is negative (no open file), and may be true or
  *   false when m_fd is non-negative (open file).
  *
  * \todo describe other attributes
  *
  * fd_set& fd_set_cpy
  *
  * - COPY of [fd_set] that will be input to select(2):  which FDs are
  *   tested, and which have data.  This is not an attribute of this
  *   class per se, but it will be declared by the outer process that
  *   contains instances of this class, and modified by the routines in
  *   this class.
  *
  * The update_status(...) and update_fds_set(...) are private functions
  * of this class, and are only called from the public interaces of this
  * class, to ensure a consistent set of values between m_fd, m_sel, and
  * the contents of the caller's [fd_set] referred to by fd_set_cpy.
  *
  * \todo describe process ID convention (<executable> -n <hbmname>)
  */
// /////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////

class HexbeatMonitor
{
public: // interfaces

    /// Constructor
    /** Ensure SIGCHLD signal handler is installed, but only once
      */
    HexbeatMonitor() { HexbeatMonitor::install_sigchld_handler(); }

    /// Check for opened or active HexbeatMonitor that matches both
    //  a given command and a given HexbeatMonitor name
    bool match(const std::string& argv0, const std::string& hbname)
    {
        return m_fd > -1 && argv0 == m_argv0 && hbname == m_hbname;
    }

    // Public read-only access to private class members
    int FD() { return m_fd; }
    std::string hbname() const { return m_hbname; }
    std::string fifo_name() const { return m_fifo_name; }
    const std::string& last_hb() const { return m_last_hb; }

    // Public read/write access to private class members (properties)
    int max_restarts_get() const { return m_restart_max; }
    void max_restarts_set(int val) { m_restart_max = val; }

    bool pending_close_get() const { return m_pending_close; }
    void pending_close_set(bool tf)
    {
        m_pending_close = tf && (m_fd > -1);
    }

    void output_redirect_set(void (*output_redirect)(std::string))
    {
        m_output_redirect = output_redirect;
    }

    int init_dly() const { return m_init_delay; }

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Open a named FIFO, load results into HexbeatMonitor instance
    /** N.B. this is a static interface of the HexbeatMonitor class:
      *      1) The FIFO's FD is not known when this function is called
      *      2) Therefore the HexbeatMonitor instance that will contain
      *         the state of that FIFO, which instance the caller
      *         references from an array of such instances, is also not
      *         known
      *      3) Once the FIFO is opened and its valid (non-negative) FD
      *         is known, the FD is used as an index into the vhexbeats
      *         vector argument to locate the target HexbeatMonitor
      *         instance
      * \returns FD (file descriptor) of file opened
      * \arg \c argv0 is the path to the hexbeater executable
      * \arg \c hbname is the hexbeater name (command-line option -n)
      * \arg \c fd_set_cpy is an [fd_set] object that contains bits of
      *                    non-inactive hexbeater FDs
      * \arg \c nfds is the ordinal of highest set bit in fd_set_cpy
      * \arg \c vhexbeats is a vector of HexbeatMonitor's
      * \arg ... => varargs are (char*) directory components of the FIFO
      *             path, ending with a NULL
      * N.B. there is no corresponding close_fifo; the FIFO FD (md_fd)
      *      will be closed automatically when the m_fd is reset to -1
      *      via routine update_status(...)
      * \todo make errors throw exceptions
      */
    static int
    open_hexbeater(const std::string& argv0, const std::string& hbname
                  , fd_set& fd_set_cpy, int& nfds
                  , std::vector<HexbeatMonitor>& vhexbeats
                  , int init_delay
                  , va_list ap
                  )
    {
        // Build the FIFO path, open the FIFO, return on failure
        std::string fifo_name = HexbeatMonitor::build_fifo_path(hbname, ap);
        int fd = HexbeatMonitor::open_hexbeater_fifo(fifo_name, vhexbeats);
        if (fd < 0) { return -1; }

        // Initialize the instance to the [opened] operational state
        vhexbeats[fd].init_on_open(argv0, hbname, fd_set_cpy, nfds
                                  , fd, fifo_name, init_delay);

        return fd;
    }
    // /////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Find a hexbeater process, or start one via fork(2)
    /** \returns pid > 0 of found/started process, and updates FD set
      * \returns 0 if that process was already running
      * \returns -1 with errno=0 if this HexbeatMonitor is inactive
      * \returns a value < 0 if fork() fails
      * \arg \c fd_set_cpy is an [fd_set] object that contains bits of
      *                    non-inactive hexbeater FDs
      * \arg \c nfds is the ordinal of highest set bit in fd_set_cpy
      * \arg \c delay is a time offset used to initialize m_last_hb
      * - so the new heartbeater process has time to startup before
      *   sending its first hexbeat
      * \arg \c check_existing tells fork-hexbeater to check for an
      *         existing process with this instance's argv0 and hbname;
      *         if such an existing process is found, then return that
      *         existing process' pid and do not fork a new process
      */
    int
    start_hexbeater(fd_set& fd_set_cpy, int& nfds, int delay
                   ,bool check_existing=true
                   )
    {
        if (m_fd < 0) { errno = 0; return -1; }

        // Find running hexbeater or fork new hexbeater; return on error
        int pid = fork_hexbeater(delay,check_existing);
        if (pid < 0) { return pid; }

        // Update, select monitoring flag, [fd_set] bit, and nfds
        update_status(m_fd, true, fd_set_cpy, nfds);

        // Flush the FIFO and return the positive PID
        flushFIFO();
        return pid;
    }
    // /////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Stop a hexbeater process, found by m_argv0 and m_hbname
    /** \returns pid > 0 of stopped process, and updates FD set
      * \returns 0 if that process was not found
      * \returns -1 with errno=0 if this HexbeatMonitor is inactive
      * \returns -1 with errno!=0 if kill(...) throws an error
      * \arg \c fd_set_cpy is an [fd_set] object that contains bits of
      *                    non-inactive hexbeater FDs
      * \arg \c nfds is the ordinal of highest set bit in fd_set_cpy
      */
    int
    stop_hexbeater(fd_set& fd_set_cpy, int& nfds)
    {
        if (m_fd < 0) { errno = 0; return -1; }
        int pid = find_hexbeater_pid();
        if (pid < 1)
        {
            update_status(m_fd, false, fd_set_cpy, nfds);
            errno = 0;
            return 0;
        }
        int istatus = kill(pid, SIGUSR2);
        if (istatus < 0) { return -1; }
        update_status(m_fd, false, fd_set_cpy, nfds);
        return pid;
    }
    // /////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Close this HexbeatMonitor instance
    /** \returns 0 if instance is closed here
      * \returns -1 if instance was already closed
      * \arg \c fd_set_cpy is an [fd_set] object that contains bits of
      *                    non-inactive hexbeater FDs
      * \arg \c nfds is the ordinal of highest set bit in fd_set_cpy
      */
    int
    close_hexbeater(fd_set& fd_set_cpy, int& nfds)
    {
        if (m_fd < 0) { return -1; }
        // Stop hexbeater process, if one is running
        stop_hexbeater(fd_set_cpy, nfds);
        // Reset FD to -1, update caller's [fd_set], close FD
        update_status(-1, false, fd_set_cpy, nfds);
        return 0;
    }
    // /////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Check bits in fd_set, read data from FD, parse possible hexbeat
    /** \returns 0 if new hexbeat is parsed
      * \returns 0 if no newline is found
      * \returns -1 with errno!=0 if read(...) throws an error
      * \returns -1 if newline found but no hexbeat is parsed
      * \arg \c fd_set_post_select is an [fd_set] object that was
      *                            ***modified & output*** via select(2)
      */
    int
    read_hexbeater(fd_set& fd_set_post_select)
    {
        // If process is inactive, or select disabled, or null [fd_set],
        // or bit is clear in [fd_set], then do nothing
        if (m_fd < 0 || !m_sel) { return 0; }
        if (!FD_ISSET(m_fd, &fd_set_post_select)) { return 0; }

        // Append data to buffer, parse buffer, clear buffer
        return append_and_parse();
    }
    // /////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Check if latest value of hexbeat has expired wrt current time
    /** \returns false if either not started, or hexbeat is not expired
      * \returns ture if hexbeater is started and hexbeat is expired
      * \arg \c hbnow is the current time as a hexbeat string
      */
    bool
    late_hexbeat(const std::string& hbnow)
    {
        // If process is active, AND select is enabled, AND current
        // hexbeat argument exceeds last hexbeat received, then hexbeat
        // has expired
        return m_fd > -1 && m_sel && (hbnow > m_last_hb);
    }
    // /////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Find hexbeater PID by executable and hexbeater name
    /** \returns PID of matching process from /proc/ filesystem
      * N.B. this is a class-static function
      *      - Per-instance function, below, calls this function
      * The hexbeater must have been started with the command
      *   argv0 -n hexbeater_name
      * or equivalent (e.g. execvp)
      * \arg \c argv0 is the argv[0] of the process
      * \arg \c hexbeater_name is the argv[2] of the process,
      * following the -n in argv[3]
      *
      * $ echo Sanford | od -a -b -tx1 -tx4
      * 0000000   S   a   n   f   o   r   d  nl
      *         123 141 156 146 157 162 144 012
      *          53  61  6e  66  6f  72  64  0a
      *                666e6153        0a64726f
      * 0000010
      * $
      * $
      * $ od -aw32 /proc/12645/cmdline
      * 0000000   .   /   s   o   m   e   t   h   i   n   g nul   -   n nul   a   a   a   a nul
      * 0000024
      * $
      */
    static int
    find_hexbeater_pid(const std::string& argv0, const std::string& hexbeater_name)
    {
        // Open the /proc/ directory
        DIR* pdir;
        struct dirent* de;

        pdir = opendir("/proc");
        if (pdir == NULL) { exit(1); }

        int save_errno = 0;
        errno = 0;

        // Loop over entries in the /proc/ directory
        while ((de=readdir(pdir)) != NULL)
        {
            // Only look at entries that are directories with names
            // comprising digit characters
            if (de->d_type != DT_DIR) { continue; }
            char* p = de->d_name;
            int pid = 0;
            while (isdigit(*p)) { pid = (10 * pid) + *p - '0'; ++p; }
            if (p==de->d_name || *p) { continue; }

            // Open /proc/<pid>/cmdline file, read contents into
            // character buffer, close file
            std::string fncmdline("/proc/");;
            fncmdline += de->d_name;
            fncmdline += "/cmdline";
            int fdcmdline = open(fncmdline.c_str(),O_RDONLY);
            if (fdcmdline < 0) { perror(fncmdline.c_str()); errno = 0; continue; }
            char buf[4096];
            int nchars = read(fdcmdline,buf,4095);
            close(fdcmdline);

            // cmdline contents contain null terminated command-line
            // tokens, and must contain at least 7 characters:  at least
            // two for -n; at least one each for argv0 and
            // hexbeater_name; one for each null terminator
            if (nchars < 7) { errno = 0; continue; }
            buf[nchars] = '\0';

            // Step through null-terminated command-line tokens
            p = buf;
            char* pend = p + nchars;
            std::vector<std::string> hexbeater_args;
            hexbeater_args.clear();
            while (p<pend)
            {
                if (hexbeater_args.size()==0)
                {
                    if (HexbeatMonitor::matches(p,"python"))
                    {
                        p += strlen(p) + 1;
                        continue;
                    }
                }
                hexbeater_args.push_back(p);
                // If more than 3 arguments then it's not a hexbeater
                if (hexbeater_args.size()>3) { break; }
                // Advance pointer to next command-line token
                p += hexbeater_args.back().size() + 1;
                // If argv[1] is not -n, then it's not a hexbeater
                if (hexbeater_args.size()==2 && hexbeater_args[1]!="-n") { break; }
                // If argv[0] is some form of "indiserver" then break after 3 arguments
                if (hexbeater_args.size()==3 && HexbeatMonitor::is_is(argv0)) { break; }
            }

            // Eliminate non-hexbeaters
            if ( hexbeater_args.size() != 3) { continue; }
            if ( hexbeater_args[0] != argv0) { continue; }
            if ( hexbeater_args[2] != hexbeater_name) { continue; }

            // Found a hexbeater:  close dir, read PID, return PID
            closedir(pdir);
            return pid;
        }

        save_errno = errno;

        // No matching hexbeater found:  close dir; return 0 or -1
        closedir(pdir);
        errno = save_errno;
        return save_errno ? -1 : 0;
    }
    // /////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Update restart parameter
    /** \returns boolean, true if restart parameter has exceeded limit
      */
    bool
    update_restart_check_expiry()
    {
        // Increment restart parameter, return true after Nth restart
        return ++m_restart > m_restart_max && m_restart_max != 0;
    }
    // /////////////////////////////////////////////////////////////////

    friend std::ostream& operator<<(std::ostream&, const HexbeatMonitor&);

private: // Internal attributes and interfaces

    int m_fd{-1}; ///< File Descriptor (FD) of named FIFO
    // N.B.
    // - If FD is negative, then this instance represents no process
    // - If FD is non-negative, then it can be only one value, which is
    //   the offset of this HexbeatMonitor instance in an array of same

    int m_restart{0}; ///< Accumulated restart parameter (Arp)

    int m_restart_max{0}; ///< Max Arp; 0 => infinity

    bool m_sel{false}; ///< Do monitoring of this FD if m_fd > -1

    std::string m_argv0; ///< Executable of the process

    std::string m_hbname; ///< Name of the hexbeater (-n <hbname>

    int m_init_delay {10}; ///< Initial hexbeat offset

    std::string m_last_hb; ///< Most-recent 9-char heartbeat

    std::string m_fifo_name; ///< Name of the heartbeat FIFO

    std::string m_buffer{""};  ///< Accumulated heartbeat data

    bool m_pending_close{false}; ///< Mark for possible closure

    void (*m_output_redirect)(std::string){nullptr}; ///< forked outputs

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Initialization of HexbeatMonitor instance on FIFO open
    void
    init_on_open(const std::string& argv0, const std::string& hbname
                , fd_set& fd_set_cpy, int& nfds
                , int fd, std::string& fifo_name, int init_delay)
    {
        // On success, initialize instance data to opened state, not started ...
        m_init_delay = init_delay;
        m_sel = false;  // ... by leaving select monitoring off
        m_fd = fd;
        pending_close_set(false);
        update_fd_set(fd_set_cpy, nfds);
        m_argv0 = argv0;
        m_hbname = hbname;
        m_restart = 0;
        m_last_hb = "000000000";
        m_fifo_name = fifo_name;
        m_buffer.clear();
    }
    // /////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Update FD and/or select monitor flag, as well as caller's fd_set
    /** \arg \c new_fd is the desired new FD for this instance (Note 1)
      * \arg \c dosel is the new select monitoring flag (m_sel)
      * \arg \c fd_set_cpy is an [fd_set] object that contains bits of
      *                    non-inactive hexbeater FDs
      * \arg \c nfds is the ordinal of highest set bit in fd_set_cpy
      * Note 1) the value of new_fd will be assigned to instance
      *         attribute m_fd, with the following caveat:  new_fd must
      *         be EITHER the offset of this instance into the caller's
      *         array of instances, OR -1; if the instance is not
      *         inactive i.e. m_fd > -1, then new_fd will be
      */
    void
    update_status(int new_fd, bool dosel, fd_set& fd_set_cpy, int& nfds)
    {
        // Do nothing if the FD and select monitoring flag do not change
        if (new_fd==m_fd && dosel==m_sel) { return; }

        if (m_fd>-1 && new_fd!=m_fd)
        {
            // N.B. the instance has been opened (m_fd>-1), but new_fd
            //      is a new FD value, so the instance MUST transition
            //      to the inactive state, and the ONLY VALID new FD
            //      value is -1
            // - Clear select monitoring flag
            // - Clear existing [fd_set] bit and update nfds
            //   - N.B. ***before*** changing m_fd to -1
            // - Close FIFO
            // - Assign m_fd = -1, completing the transition to inactive
            m_sel = false;
            update_fd_set(fd_set_cpy, nfds);
            // ensure FIFO is closed!
            close(m_fd);
            m_fd = -1;
            pending_close_set(false);
            return;
            // \todo perhaps throw an exception if new_fd is not -1
        }

        if (dosel) { m_last_hb = time_to_hb(m_init_delay); }
        m_fd = new_fd;
        pending_close_set(false);
        m_sel = dosel;
        // Keep fd_set in synchrony with FD and select monitoring status
        update_fd_set(fd_set_cpy, nfds);
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// Turn select(2) monitoring on or off for this FD
    /** \arg \c fd_set_cpy is an [fd_set] object that contains bits of
      *                    non-inactive hexbeater FDs
      * \arg \c nfds is the ordinal of highest set bit in fd_set_cpy
      */
    void
    update_fd_set(fd_set& fd_set_cpy, int& nfds)
    {
        // Process is inactive, ensure its select monitoring is disabled
        if (m_fd < 0)
        {
            m_sel = false;
            pending_close_set(false);
            return;
        }

        if (m_sel)
        {
            // Turn select monitoring on and return
            FD_SET(m_fd, &fd_set_cpy);
            if (nfds<=m_fd) { nfds = m_fd + 1; }
            return;
        }

        // Turn select monitoring off:
        FD_CLR(m_fd, &fd_set_cpy);
        if (nfds == (m_fd+1))
        {
            // This was last monitored FD:  get next lower monitored FD
            while ((nfds>0) && !FD_ISSET(nfds-1,&fd_set_cpy)) { nfds--;}
        }
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// SIGCHLD signal handler
    /** Ignores SIGCHLD signals while preventing zombies
      */
    static void
    hbm_sigchld_handler(int sig)
    {
        static_cast<void>(sig);
        int saved_errno = errno;
        while (waitpid((pid_t)(-1), 0, WNOHANG) > 0) {}
        errno = saved_errno;
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// Install SIGCHLD signal handler
    /** Routine is static to ensure this happens only once
      */
    static void
    install_sigchld_handler()
    {
        static int singleton{0};
        if (singleton) { return; }

        struct sigaction sa;
        sa.sa_handler = &HexbeatMonitor::hbm_sigchld_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART | SA_NOCLDSTOP | SA_NOCLDWAIT;
        if (sigaction(SIGCHLD, &sa, 0) == -1)
        {
            perror("Error in HexbeatMonitor:: SIGCHLD handler install");
            exit(1);
        }
        singleton = 1;
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // /////////////////////////////////////////////////////////////////
    /// Find starting position of last 9dnths in buffer
    /** \returns starting position of last 9dnths in buffer
      * \returns std::string::npos if no 9ndths found,
      *   because
      *   - either no newline found,
      *   - or 9 hexadecimal digits did not precede the last newline
      */
    static std::size_t
    hex9_nlterminated(const std::string& buffer, std::size_t& inl)
    {
        // Look for last newline delimiter with at least 9 preceding
        // characters
        inl = buffer.rfind('\n');
        if (inl==std::string::npos || inl<9) { return std::string::npos; }

        // Check that preceding 9 characters are all hexadecimal digits
        for (std::size_t ipos=inl-9; ipos<inl; ++ipos)
        {
            if (!isxdigit(buffer[ipos])) { return std::string::npos; }
        }
        return inl-9;
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// Build FIFO pathname
    static std::string
    build_fifo_path(const std::string& hbname, va_list ap)
    {

        // Join varargs to combine the FIFO directory paths, with slashes
        char* cptr;
        std::string fifo_name("");
        while((cptr=va_arg(ap,char*)))
        {
            fifo_name += cptr;
            // Append trailing slash if not already present
            if (*cptr && fifo_name.back() != '/') { fifo_name += "/"; }
        }

        // Append name and .hb suffix to complete named FIFO
        fifo_name += hbname;
        fifo_name += ".hb";

        return fifo_name;
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    static int
    open_hexbeater_fifo(const std::string& fifo_name
                       , std::vector<HexbeatMonitor>& vhexbeats
                       )
    {
        // Open FIFO read-write and non-blocking; create FIFO if needed
        int fd = open(fifo_name.c_str(),O_RDWR|O_NONBLOCK|O_CLOEXEC);

        int istat{0};

        if (fd < 0 && errno == ENOENT)
        {
            // File does not exist:  create FIFO; re-attempt open
            mode_t prev_mode;
            errno = 0;
            prev_mode = umask(0);
            istat = mkfifo(fifo_name.c_str(), S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP);
            prev_mode = umask(prev_mode);
            if (istat < 0) { return -1; }
            fd = open(fifo_name.c_str(),O_RDWR|O_NONBLOCK|O_CLOEXEC);
        }
        if (fd < 0) { return -1; }

        //  Test that opened FD is a FIFO
        struct stat st;
        istat  = fstat(fd,&st);
        bool not_FIFO = (istat < 0) || !S_ISFIFO(st.st_mode);

        // Error if either file is not a FIFO, or FD is already in use
        if (not_FIFO || (vhexbeats[fd].m_fd > -1))
        {
            close(fd);
            errno = EEXIST;
            return -1;
        }
        return fd;
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// Find this instance's hexbeater PID by argv[0] and hexbeater name
    /** Use the static HexbeatMonitor::find_hexbeater_pid below
      * \returns PID of matching process from /proc/ filesystem
      */
    int
    find_hexbeater_pid()
    {
         if (m_fd < 0) { return 0; }
         return HexbeatMonitor::find_hexbeater_pid(m_argv0, m_hbname);
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// Fork the hexbeater, perhaps only if it is not already running
    int
    fork_hexbeater(int delay, bool check_existing)
    {
        static_cast<void>(delay);
        // If directed to check for a hexbeater via m_argv0 and m_hbname
        // and such a hexbeater is found, then return its PID ...
        int pid{check_existing ? find_hexbeater_pid() : 0};
        if (pid > 0) { return pid; };

        // ... Otherwise fork, ...
        pid = fork();
        if (pid != 0)
        {
            // Parent fork:  on success (pid > 0) add delay to current
            // time to initialize last hexbeat; on error (pid < 0) do
            // nothing extra; on either return that pid
            if (pid > 0) { m_last_hb = time_to_hb(m_init_delay); }
            return pid;
        } // fork failed (<0) or parent (>0)

        // Child fork:  pid == 0

        int save_errno{errno};
        int ipgstat = setpgid(0,0);
        if (ipgstat)
        {
            std::cerr
            << ipgstat << "=setpgid(0["
            << m_argv0 << " -n " << m_hbname
            << "],0)"
            << "; " << errno << "(" << strerror(errno) << ")=errno"
            << "; continuing ..."
            << std::endl;
            errno = save_errno;
        }

        if (m_output_redirect)
        {
            errno = 0;
            m_output_redirect(m_hbname);
        }

        // ... And then exec, the hexbeater
        // Child:  <argv0> -n <name>
        const char* argv0 = m_argv0.c_str();
        const char* name = m_hbname.c_str();
        int e = execlp(argv0, argv0, "-n", name, (char*) NULL);
        // \todo pass any error back to the parent (pipe?)
        if (e) exit(-11);
        return -1;   // Execution should never get here
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// Flush and discard data from this instance's open FIFO
    /** Called by public interface start_hexbeater(...) above
      */
    void
    flushFIFO()
    {
        if (m_fd < 0) { return; }

        struct timeval tv;
        char c1024[1024];
        fd_set fdset;
        FD_ZERO(&fdset);
        while (1)
        {
            tv.tv_sec = tv.tv_usec = 0;
            FD_SET(m_fd, &fdset);
            if (select(m_fd+1, &fdset,0,0, &tv) != 1) { break; }
            if (read(m_fd, c1024, 1024) < 1) { break; }
        }
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// Append data to instance buffer, parse for hexbeat, clean buffer
    /** Called by public interface read_hexbeater(...) above
      * \returns -1 on error (e.g. if lenc10 is negative)
      * \returns 0 for either successful, or insufficient data to, parse
      */
    int
    append_and_parse()
    {
        // Read data repeatedly and append onto buffer until read error
        ssize_t lenc10;
        do
        {
            char c10[10];
            errno = 0;
            // read(2) will throw an EAGAIN/EWOULDBLOCK errno eventually
            // \todo convince myself this cannot loop forever?
            lenc10 = ::read(m_fd,c10,10);
            if (lenc10 > 0)
            {
                // Append read input data to buffer
                m_buffer.append(c10, lenc10);
                int L = m_buffer.size();
                // Erase any data too old to be of use; 20 allows for
                // full 9-char hexbeat + NL, plus 10 more chars w/o NL
                if (L > 20) { m_buffer.erase(0, L-20); }
            }
        } while ( lenc10 > -1);

        // Read error
        if (errno != EWOULDBLOCK && errno != EAGAIN) { return -1; }

        // Find locations in buffer of start of heartbeat and/or of '\n'
        std::size_t inl;
        std::size_t hex9 = HexbeatMonitor::hex9_nlterminated(m_buffer, inl);

        // Valid hexbeat found:  store it; erase data through newline
        if (hex9!=std::string::npos)
        {
            m_last_hb = m_buffer.substr(hex9, 10);
            m_buffer.erase(0,inl+1);
            return 0;
        }

        size_t L = m_buffer.size();

        // No newline:  erase any obsolete buffer; wait for more data
        if (inl==std::string::npos)
        {
            if (L > 9) { m_buffer.erase(0,L-9); }
            return 0;
        }

        // Newline but no hexbeat:  erase obsolete data or thru newline
        if ((L-9) > inl)
        {
            m_buffer.erase(0,L-9);
        }
        else
        {
            m_buffer.erase(0,inl+1);
        }

        return -1;
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// Check if argv0 is some form of indiserver, else false
    /** \returns true if argv0 is some form of indiserver
      * \returns false otherwise
      */
    static bool
    is_is(const std::string& argv0)
    {
        return HexbeatMonitor::matches(argv0, "indiserver");
    }
    // ////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////
    // ////////////////////////////////////////////////////////////////
    /// Check if arg ends in [/]<match> or [/]<match>N or [/]<match>N.N
    /** \returns true if arg is some form of match
      * \returns false otherwise
      */
    static bool
    matches(const std::string& arg, const char* match)
    {
        if (!match) { return false; }   // match is null pointer
        if (!*match) { return false; }  // match is empty string
        char* pm = ((char*)match) + strlen(match);  // pointer past end

        // checkseq:  0=>possible NN.NN suffix; 1=>chars in match;
        //            where Ns are decimal digits
        // - Do not ignore N.N in arg if match ends in N or .,
        int checkseq{(pm[-1]=='.' || isdigit(pm[-1])) ? 1 : 0};

        for (std::string::const_reverse_iterator rit=arg.rbegin()
            ; rit!=arg.rend()
            ; ++rit
            )
        {
            switch(checkseq)
            {
            case 0:
                // Ignore N/./.N/N./N.N suffix in arg, if present
                if ('.'==*rit or isdigit(*rit)) { break; }
                // Char is not a decimal digit or a '.' - drop through
                checkseq = 1;
                /* falls through */

            case 1:
                // If first position in match was reached but there are
                // more chars in arg, then if current char in arg is a
                // slash, then arg matches, else arg does not match
                if (match == pm) { return '/'==*rit; }

                // Check next earlier position in match
                --pm;
                if (*rit != *pm) { return false; }
                break;
            }
        }
        // All characters in arg have been checked and match the
        // corresponding characters in match, so return true if there
        // are no more unchecked characters in match
        return match==pm;
    }
    // ////////////////////////////////////////////////////////////////
};
std::ostream& operator<<(std::ostream& os, const HexbeatMonitor& hbm)
{
    os
    << "<HexbeatMonitor \""
    << hbm.m_argv0 << " -n " << hbm.hbname() << "\""
    << "; FD=" << hbm.m_fd
    << "; FIFO=" << hbm.fifo_name()
    << "; lastHB=" << hbm.last_hb().substr(0,9)
    << "; restart=" << hbm.m_restart
    << ">"
    ;
    return os;
}
