#include <set>
#include "HexbeatMonitor.hpp"

#ifndef MACRO_FD_SETSIZE
#define MACRO_FD_SETSIZE FD_SETSIZE
#endif

template <int HBR_FD_SETSIZE = MACRO_FD_SETSIZE>
class resurrectorT
{
private:
    int m_nfds{0}; ///< ordinal of highest set bit in m_fdset_cpy

    fd_set m_fdset_cpy; ///< ***COPY*** of active set bits of hexbeat file descriptors;
                        ///  N.B. DO NOT PASS THIS fd_set to select(2)!

    bool m_logging{false};
    bool m_verbose{false};

    std::set<int> m_fds; ///< FDs of all opened hexbeaters in m_hbmarr

    std::vector<HexbeatMonitor> m_hbmarr{std::vector<HexbeatMonitor>(HBR_FD_SETSIZE)}; ///< vector of hexbeat monitors, active or not

public:
    /// Constructor
    /** Ensure FD set copy is all zeros, and set<int> of FD is empty
      * Assign output redirect in all Hexbeater instances
      */
    resurrectorT(void (*output_redirect)(std::string)=nullptr) : m_nfds(0), m_fds({})
    {
        FD_ZERO(&m_fdset_cpy);
        for (std::vector<HexbeatMonitor>::iterator it = m_hbmarr.begin(); it != m_hbmarr.end(); ++it)
        {
            it->output_redirect_set(output_redirect);
        }
    }

    /// Assign all HexbeatMonitors' pending_close states true or false
    void
    pending_close_all_set(const bool tf)
    {
        for (std::vector<HexbeatMonitor>::iterator it = m_hbmarr.begin(); it != m_hbmarr.end(); ++it)
        {
            it->pending_close_set(tf);
        }
    }

    /// Assign specific HexbeatMonitors' pending_close states
    void
    pending_close_all_set_on_match(const bool tf, std::string& argv0, std::string& hbname)
    {
        for (std::vector<HexbeatMonitor>::iterator it = m_hbmarr.begin(); it != m_hbmarr.end(); ++it)
        {
            if (it->match(argv0, hbname)) { it->pending_close_set(tf); }
        }
    }

    /// Close any HexbeatMonitors that remain pending to be closed
    void
    pending_close_all_close()
    {
        for (std::vector<HexbeatMonitor>::iterator it = m_hbmarr.begin(); it != m_hbmarr.end(); ++it)
        {
            if (it->pending_close_get())
            {
                it->close_hexbeater(m_fdset_cpy, m_nfds);
                it->pending_close_set(false); // Probably not necessary
            }
        }
    }

    /// Run one select/read/check/restart cycle
    /** \arg \c tv - struct timeval of select(2) timeout
      * \todo ensure buffer is empty after read_hexbeater
      */
    void
    srcr_cycle(struct timeval& tv)
    {
        // Make current time a hexbeat to compare with incoming hexbeats
        // - The 0 is the offset from the current time
        // - Incoming hexbeats will have a positive offset relative to
        //   the time they are created; that offset will be in effect a
        //   timeout for the hexbeater process that sends it.  By using
        //   this approach, the timeout can be specific to the hexbeater
        //   process, and this resurrector process does not need to keep
        //   track of the individual hexbeaters' timeouts.
        std::string hbnow{time_to_hb(0)};

        if (m_verbose) { std::cerr << hbnow; } 

        // Copy current class member of [fd_set] bits to local [fd_set],
        // then call select(2) to determine if any hexbeat FDs have data
        errno = 0;
        fd_set lcl_fdset{m_fdset_cpy};
        int iselect = select(m_nfds, &lcl_fdset,0,0, &tv);

        // Handle select(2) error
        if (iselect < 0)
        {
            // Log and pause for errors; respond to signals immediately
            if (errno != EINTR)
            {
#               define ISELSTR(x) #x
#               define ISELSTRMACRO(x) ISELSTR(x)
                perror("select(2) error:  " __FILE__ "; " ISELSTRMACRO(__LINE__));
                // Pause for about a second
                struct timeval tv{0,999999};
                select(0, 0,0,0, &tv);
            }
            return;
        }

        // Loop on the active FDs, each representing an active hexbeater
        for (std::vector<HexbeatMonitor>::iterator it = m_hbmarr.begin(); it != m_hbmarr.end(); ++it)
        {
            // Read any hexbeat data from the hexbeater FIFO, compare
            // the local hexbeat (hbnow, above) to the latest received
            // hexbeat, and if that received hexbeat has not timed out,
            // then continue to the the next hexbeater
            it->read_hexbeater(lcl_fdset);
            bool islate = it->late_hexbeat(hbnow);
            if ((m_verbose
                || (m_logging && islate)
                )
             && it->FD() > -1 && it->FD() < HBR_FD_SETSIZE)
            {
                fd_to_stream(std::cerr, it->FD());
                std::cerr << (islate ? ":late" : ":okay") << std::endl;
            }
            if (!islate) { continue; }

            // To here, the current hexbeater, represented by the fd
            // from m_fds and hbm, has timed out.  Attempt to stop the
            // hexbeater process, in case it is locked up, then wait
            // for a tenth of a second, then re-start a new hexbeater
            // process.
            // Pass the the [fd_set] bits (m_fdset_cpy) and the range of
            // bits (m_nfds) for the active hexbeaters to be updated
            // with each call
            if (it->update_restart_check_expiry())
            {
                // If the restart parameter has expired, issue a message
                // and close that hexbeater FIFO, which also stops
                // further restarts of this hexbeater process
                std::cerr << "[resurrector giving up on " << *it
                          << " at " << time_to_hb(0).substr(0,9) << "]\n";
                it->close_hexbeater(m_fdset_cpy, m_nfds);
                continue;
            }
            if (m_logging)
            {
                std::cerr << "[resurrector stopping " << *it
                      << " at " << time_to_hb(0).substr(0,9) << "]\n";
            }
            it->stop_hexbeater(m_fdset_cpy, m_nfds);
            struct timeval tv{0,99999};
            select(0, 0,0,0, &tv);
            if (m_logging)
            {
                std::cerr << "[resurrector re-starting " << *it
                          << " at " << time_to_hb(0).substr(0,9)
                          << "]\n";
            }
            it->start_hexbeater(m_fdset_cpy, m_nfds,it->init_dly()
                               ,false);
        }
    }

    /// Add one hexbeat monitor
    int
    open_hexbeater(int init_delay, const std::string& argv0, const std::string& hbname
                  , ...)
    {
        // Exit with error if hbname is already present in m_hbmarr
        if (find_hbm_by_name(hbname) > -1) { errno = EEXIST; return -1; }

        // Initialize varargs; open new FIFO; clean up varargs
        va_list ap; va_start(ap, hbname);
        int newfd = HexbeatMonitor::open_hexbeater(
                    argv0, hbname, m_fdset_cpy, m_nfds, m_hbmarr
                    , init_delay
                    , ap);
        va_end(ap);

        if (newfd < 0)
        {
            if (m_logging)
            {
                perror(("Hexbeater::open_hexbeater"
                         "(" + argv0
                       + "," + hbname
                       + ") failed to open FIFO"
                       ).c_str()
                       );
            }
            return -1;
        }
        // Add FD to set
        m_fds.insert(newfd);
        return newfd;
    }

    /// Close a hexbeat monitor
    int
    close_hexbeater(int fd)
    {
        if (m_fds.find(fd) == m_fds.end()) { return -1; }
        m_hbmarr[fd].close_hexbeater(m_fdset_cpy, m_nfds);
        return 0;
    }

    /// Start one hexbeat monitor
    int
    start_hexbeater(int fd, int max_restarts=0)
    {
        if (m_fds.find(fd) == m_fds.end()) { return -1; }
        m_hbmarr[fd].max_restarts_set(max_restarts);
        return
            m_hbmarr[fd].start_hexbeater(m_fdset_cpy, m_nfds
            , m_hbmarr[fd].init_dly());
    }

    /// Return FIFO name of HexbeatMonitor element fd of m_hbmarr array
    /** \returns "" if HexbeatMonitor state is inactive
      * \returns HexbeatMonitor FIFO name if state is opened or started
      * \arg -c fd - offset into m_hbmarr
      * Value will be either "" if HexbeatMonitor does not have an
      * associated hexbeat channel (e.g. FIFO /.../<hbname>.hb),
      * or it will be m_fifo_name from that HexbeatMonitor instance
      */
    std::string
    fifo_name_hbm(int fd)
    {
        if (FDhbm(fd) < 0) { return std::string(""); }
        return m_hbmarr[fd].fifo_name();
    }

    /// Return FD of HexbeatMonitor element fd of m_hbmarr array
    /** \returns -999999 if fd is invalid
      * \returns -1 if HexbeatMonitor state is inactive
      * \returns fd if HexbeatMonitor state is opened or started
      * \arg -c fd - offset into m_hbmarr
      * Value will be either negative or fd
      */
    int
    FDhbm(int fd)
    {
        if (fd < 0 || fd >= HBR_FD_SETSIZE) { return -999999; }
        return m_hbmarr[fd].FD();
    }

    /// Find FD from set m_fds of item in m_hbmarray with name hbmname
    /** Value will be either -1 if no HexbeatMonitor has that name
      * or it will be the FD of the HexbeatMonitor with that name
      */
    int
    find_hbm_by_name(const std::string& hbmname)
    {
        for (auto fd : m_fds)
        {
            if (FDhbm(fd) < 0) { continue; }
            if (hbmname == m_hbmarr[fd].hbname()) { return fd; }
        }
        return -1;
    }

    /// Find PID of process of element fd of m_hbmarr array
    int
    find_pid(int fd)
    {
        if (fd < 0 || fd >= HBR_FD_SETSIZE) { return -999999; }
        return m_hbmarr[fd].find_hexbeater_pid();
    }

    /// Debugging
    int get_fd_setsize() { return HBR_FD_SETSIZE; }
    int calcsize() { return (sizeof m_hbmarr) / (sizeof m_hbmarr[0]); }

    void fd_to_stream(std::ostream& os, int fd)
    {
        if (fd > -1 && fd < HBR_FD_SETSIZE) { os << m_hbmarr[fd]; }
    }

    /// Logging
    void set_resurr_logging() { m_logging = true; }
    void clr_resurr_logging() { m_logging = false; }
    void set_resurr_verbose_logging() { m_verbose = m_logging = true; }
    void clr_resurr_verbose_logging() { m_verbose = false; }
};
