#include <set>
#include "HexbeatMonitor.hpp"

#ifndef MACRO_FD_SETSIZE
#define MACRO_FD_SETSIZE FD_SETSIZE
#endif

template <int HBR_FD_SETSIZE = MACRO_FD_SETSIZE>
class resurrectorT
{
private:
    fd_set m_fdset_copy; //< ***COPY*** of active set of hexbeat file descriptors;
                     //  N.B. DO NOT PASS THIS fd_set to select(2)!

    int m_nfds{0}; //< 1 + highest value of hexbeat file descriptors in m_fdset_copy

    HexbeatMonitor m_hbmarr[HBR_FD_SETSIZE]; //< array of hexbeat monitors, active or not

    //std::set<int> m_fds[HBR_FD_SETSIZE]; //< FDs of all hexbeaters in m_hbmarr
    std::set<int> m_fds; //< FDs of all hexbeaters in m_hbmarr

public:
    /// Constructor
    /** Ensures FD set copy is empty
      */
    resurrectorT(void) : m_nfds(0) { FD_ZERO(&m_fdset_copy); }

    std::vector<int>
    one_cycle()
    {
        std::vector<int> rtn;
        fd_set lcl_fdset{m_fdset_copy};
        struct timeval tv{1,0};
        int iselect = select(m_nfds, m_nfds > 0 ? &lcl_fdset : (fd_set*)0,0,0, &tv);
        if (iselect < 0)
        {
            tv.tv_sec = 1;
            select(0, 0,0,0, &tv);
            FD_ZERO(&lcl_fdset);
        }

        std::string current_hb = time_to_hb(0);

        rtn.clear();

        for (auto ifd : m_fds)
        {
            m_hbmarr[ifd].read_fifo(&lcl_fdset);
            if (current_hb > m_hbmarr[ifd].last_hb()) { rtn.push_back(ifd); }
        }
        return rtn;
    }

    /// Add one hexbeat monitor
    int
    add_hexbeat(std::string argv0, std::string hbname, ...)
    {
        // Exit with error if hbname is already present in m_hbmarr
        if (find_hbm_by_name(hbname) > -1) { errno = EEXIST; return -1; }

        // Initialize varargs
        va_list ap;
        va_start(ap, hbname);

        // Open new FIFO
        int newfd = HexbeatMonitor::open_hexbeater(argv0, hbname
                                                  , &m_fdset_copy
                                                  , m_nfds
                                                  , m_hbmarr, ap
                                                  );

        // Clean up varargs 
        va_end(ap);

        if (newfd < 0) { return -1; }

        // Add FD to set
        m_fds.insert(newfd);
        return newfd;
    }

    /// Return FIFO name of element ihbm of m_hbmarr array
    /** Value will be either "" if HexbeatMonitor does not have an
      * associated hexbeat channel (e.g. FIFO /.../<hbname>.hb),
      * or it will be m_fifo_name from that HexbeatMonitor  instance
      */
    std::string
    fifo_name_hbm(int ihbm)
    {
        if (FDhbm(ihbm) < 0) { return std::string(""); }
        return m_hbmarr[ihbm].fifo_name();
    }

    /// Return FD of element ihbm of m_hbmarr array
    /** Value will be either -1 if HexbeatMonitor does not have an
      * associated hexbeat channel (e.g. FIFO /.../<hbname>.hb),
      * or it will be ihbm
      */
    int
    FDhbm(int ihbm)
    {
        if (ihbm < 0 || ihbm >= HBR_FD_SETSIZE) { return -999999; }
        return m_hbmarr[ihbm].FD();
    }

    /// Find FD from set m_fds of item in m_hbmarray with name hbmname
    /** Value will be either -1 if no HexbeatMonitor has that name
      * or it will be the FD of the HexbeatMonitor with that name
      */
    int
    find_hbm_by_name(const std::string hbmname)
    {
        for (auto it : m_fds)
        {
            if (FDhbm(it) < 0) { continue; }
            if (hbmname == m_hbmarr[it].hbname()) { return it; }
        }
        //for (auto it = m_fds.begin(); it != m_fds.end(); ++it)
        //{
        //    if (FDhbm(*it) < 0) { continue; }
        //    if (hbmname == m_hbmarr[*it].hbname()) { return *it; }
        //}
        return -1;
    }

    int
    find_pid(int ihbm)
    {
        if (ihbm < 0 || ihbm >= HBR_FD_SETSIZE) { return -999999; }
        return m_hbmarr[ihbm].find_hexbeater_pid();
    }

    /// Debugging
    int get_fd_setsize() { return HBR_FD_SETSIZE; }
    int calcsize() { return (sizeof m_hbmarr) / (sizeof m_hbmarr[0]); }
};
