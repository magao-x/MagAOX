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

    int m_delay{10}; ///< Initial offset of hexbeat value when starting hexbeaters

    HexbeatMonitor m_hbmarr[HBR_FD_SETSIZE]; ///< array of hexbeat monitors, active or not

    std::set<int> m_fds; ///< FDs of all opened hexbeaters in m_hbmarr

public:
    /// Constructor
    /** Ensures FD set copy is all zeros, and set<int> of FD is empty
      */
    resurrectorT(void) : m_nfds(0), m_fds({})
    {
        FD_ZERO(&m_fdset_cpy);
    }

    /// Run one select/read/check/restart cycle
    /** \returns std::vector<int> of FDs with expired hexbeat timestamps
      * \arg \c ptv - pointer to struct timeval of select(2) timeout
      * \todo ensure buffer is empty after read_hexbeater
      */
    void
    srcr_cycle(struct timeval *ptv)
    {
        errno = 0;
        fd_set lcl_fdset{m_fdset_cpy};
        int iselect = select(m_nfds, &lcl_fdset,0,0, ptv);

        if (iselect < 0)
        {
            // On select error, pause for 0.999999s
            struct timeval tv{0,999999};
            select(0, 0,0,0, &tv);
            return;
        }

        std::string hbnow = time_to_hb(0);

        std::vector<int> expired_fds{};

        for (auto fd : m_fds)
        {
            HexbeatMonitor* phb = m_hbmarr + fd;
            phb->read_hexbeater(&lcl_fdset);
            if (!phb->late_hexbeat(hbnow)) { continue; }
            // \todo ensure read buffer is empty
            phb->stop_hexbeater(&m_fdset_cpy, m_nfds);
            struct timeval tv{0,99999};
            select(0, 0,0,0, &tv);
            phb->start_hexbeater(&m_fdset_cpy, m_nfds,m_delay);
        }
    }

    /// Add one hexbeat monitor
    int
    open_hexbeater(std::string argv0, std::string hbname, ...)
    {
        // Exit with error if hbname is already present in m_hbmarr
        if (find_hbm_by_name(hbname) > -1) { errno = EEXIST; return -1; }

        // Initialize varargs; open new FIFO; clean up varargs 
        va_list ap; va_start(ap, hbname);
        int newfd = HexbeatMonitor::open_hexbeater
                    (argv0, hbname, &m_fdset_cpy, m_nfds, m_hbmarr, ap);
        va_end(ap);

        if (newfd < 0) { return -1; }

        // Add FD to set
        m_fds.insert(newfd);
        return newfd;
    }

    /// Start one hexbeat monitor
    int
    start_hexbeater(int fd)
    {
        if (m_fds.find(fd) == m_fds.end()) { return -1; }
        return
            m_hbmarr[fd].start_hexbeater(&m_fdset_cpy, m_nfds, m_delay);
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
    find_hbm_by_name(const std::string hbmname)
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
};
