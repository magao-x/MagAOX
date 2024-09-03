/* INDI Server for protocol version 1.7.
 * Copyright (C) 2007 Elwood C. Downey <ecdowney@clearskyinstitute.com>
                 2013 Jasem Mutlaq <mutlaqja@ikarustech.com>
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

 * argv lists names of Driver programs to run or sockets to connect for Devices.
 * Drivers are restarted if they exit or connection closes.
 * Each local Driver's stdin/out are assumed to provide INDI traffic and are
 *   connected here via pipes. Local Drivers' stderr are connected to our
 *   stderr with date stamp and driver name prepended.
 * We only support Drivers that advertise support for one Device. The problem
 *   with multiple Devices in one Driver is without a way to know what they
 *   _all_ are there is no way to avoid sending all messages to all Drivers.
 * Outbound messages are limited to Devices and Properties seen inbound.
 *   Messages to Devices on sockets always include Device so the chained
 *   indiserver will only pass back info from that Device.
 * All newXXX() received from one Client are echoed to all other Clients who
 *   have shown an interest in the same Device and property.
 *
 * 2017-01-29 JM: Added option to drop stream blobs if client blob queue is
 * higher than maxstreamsiz bytes
 *
 * Implementation notes:
 *
 * We fork each driver and open a server socket listening for INDI clients.
 * Then forever we listen for new clients and pass traffic between clients and
 * drivers, subject to optimizations based on sniffing messages for matching
 * Devices and Properties. Since one message might be destined to more than
 * one client or device, they are queued and only removed after the last
 * consumer is finished. XMLEle are converted to linear strings before being
 * sent to optimize write system calls and avoid blocking to slow clients.
 * Clients that get more than maxqsiz bytes behind are shut down.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // needed for siginfo_t and sigaction
#endif

#include "config.h"

#include "fq.h"
#include "indiapi.h"
#include "indidevapi.h"
#include "lilxml.h"

#include <zlib.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>

#include "open_named_fifo.h"

#define INDIPORT      7624    /* default TCP/IP port to listen */
#define REMOTEDVR     (-1234) /* invalid PID to flag remote drivers */
#define LOCALDVR      (-2468) /* invalid PID to flag local drivers */
#define MAXSBUF       512
#define MAXRBUF       49152 /* max read buffering here */
#define MAXWSIZ       49152 /* max bytes/write */
#define SHORTMSGSIZ   2048  /* buf size for most messages */
#define DEFMAXQSIZ    128   /* default max q behind, MB */
#define DEFMAXSSIZ    5     /* default max stream behind, MB */
#define DEFMAXRESTART 0     /* default max restarts */

#ifdef OSX_EMBEDED_MODE
#define LOGNAME  "/Users/%s/Library/Logs/indiserver.log"
#define FIFONAME "/tmp/indiserverFIFO"
#endif

/* associate a usage count with queuded client or device message */
typedef struct
{
    int count;         /* number of consumers left */
    unsigned long cl;  /* content length */
    char *cp;          /* content: buf or malloced */
    char buf[SHORTMSGSIZ];    /* local buf for most messages */
} Msg;

/* device + property name */
typedef struct
{
    char dev[MAXINDIDEVICE];
    char name[MAXINDINAME];
    BLOBHandling blob; /* when to snoop BLOBs */
} Property;

/* record of each snooped property
typedef struct {
    Property prop;
    BLOBHandling blob;
} Property;
*/

struct
{
    const char *name; /* Path to FIFO for dynamic startups & shutdowns of drivers */
    int fd;
    //FILE *fs;
} fifo;

/* info for each connected client */
typedef struct
{
    int active;         /* 1 when this record is in use */
    Property *props;    /* malloced array of props we want */
    int nprops;         /* n entries in props[] */
    int allprops;       /* saw getProperties w/o device */
    BLOBHandling blob;  /* when to send setBLOBs */
    int s;              /* socket file descriptor (FD) of this client */
    LilXML *lp;         /* XML parsing context */
    FQ *msgq;           /* Msg queue */
    unsigned int nsent; /* bytes of current Msg sent so far */
    gzFile gzfird;      /* zlib gzread proxy for FD */
    gzFile gzfiwr;      /* zlib gzwrite proxy for FD) */
    int gzwchk;         /* Allow for one compression check */
} ClInfo;
static ClInfo *clinfo; /*  malloced pool of clients */
static int nclinfo;    /* n total (not active) */

/* info for each connected driver */
typedef struct strDvrInfo
{
    char name[MAXINDINAME]; /* persistent name */
    char envDev[MAXSBUF];
    char envConfig[MAXSBUF];
    char envSkel[MAXSBUF];
    char envPrefix[MAXSBUF];
    char host[MAXSBUF];
    int port;
    char **dev;         /* device served by this driver */
    int ndev;           /* number of devices served by this driver */
    int active;         /* 1 when this record is in use */
    Property *sprops;   /* malloced array of props we snoop */
    int nsprops;        /* n entries in sprops[] */
    int pid;            /* process id or REMOTEDVR if remote */
    int rfd;            /* read pipe fd */
    int wfd;            /* write pipe fd */
    int restarts;       /* times process has been restarted */
    LilXML *lp;         /* XML parsing context */
    FQ *msgq;           /* Msg queue */
    unsigned int nsent; /* bytes of current Msg sent so far */
    struct strDvrInfo* pNextToRestart; /* next to restart, or NULL */
    int restartDelayus; /* Microseconds before next restart attempt */
    gzFile gzfird;      /* zlib gzread proxy for FD; remote dvr only */
    gzFile gzfiwr;      /* zlib gzwrite proxy for FD; remote dvr only */
} DvrInfo, *pDvr, **ppDvr;
static DvrInfo *dvrinfo; /* malloced array of drivers */
static int ndvrinfo;     /* n total */

static char *me;                                       /* our name */
#define Mus 1000000     /* Microseconds per second */
#define SELECT_WAITs 1  /* Select wait, seconds */
static pDvr pRestarts;  /* linked list of drivers to restart */

static int port = INDIPORT;                            /* public INDI port */
static int verbose;                                    /* chattiness */
static int use_is_zlib = 0;                            /* inter-INDI server zlib compression */
static int lsocket;                                    /* listen socket */
static char *ldir;                                     /* where to log driver messages */
static int maxqsiz       = (DEFMAXQSIZ * 1024 * 1024); /* kill if these bytes behind */
static int maxstreamsiz  = (DEFMAXSSIZ * 1024 * 1024); /* drop blobs if these bytes behind while streaming*/
static int maxrestarts   = DEFMAXRESTART;
static int terminateddrv = 0;

/* fill s with current UT string.
 * if no s, use a static buffer
 * return s or buffer.
 * N.B. if use our buffer, be sure to use before calling again
 */
static char *indi_tstamp(char *s)
{
    static char sbuf[64];
    struct tm *tp;
    time_t t;

    time(&t);
    tp = gmtime(&t);
    if (!s)
        s = sbuf;
    strftime(s, sizeof(sbuf), "%Y-%m-%dT%H:%M:%S", tp);
    return (s);
}

/* log when then exit */
static void Bye()
{
    fprintf(stderr, "%s: good bye\n", indi_tstamp(NULL));
    exit(1);
}

/* record we have started and our args */
static void logStartup(int ac, char *av[])
{
    int i;

    fprintf(stderr, "%s: startup: ", indi_tstamp(NULL));
    for (i = 0; i < ac; i++)
        fprintf(stderr, "%s ", av[i]);
    fprintf(stderr, "\n");
}

/* print usage message and exit (2) */
static void usage(void)
{
    fprintf(stderr, "Usage: %s [options] driver [driver ...]\n", me);
    fprintf(stderr, "Purpose: server for local and remote INDI drivers\n");
    fprintf(stderr, "INDI Library: %s\nCode %s. Protocol %g.\n", CMAKE_INDI_VERSION_STRING, GIT_TAG_STRING, INDIV);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, " -l d     : log driver messages to <d>/YYYY-MM-DD.islog\n");
    fprintf(stderr, " -m m     : kill client if gets more than this many MB behind, default %d\n", DEFMAXQSIZ);
    fprintf(stderr,
            " -d m     : drop streaming blobs if client gets more than this many MB behind, default %d. 0 to disable\n",
            DEFMAXSSIZ);
    fprintf(stderr, " -p p     : alternate IP port, default %d\n", INDIPORT);
    fprintf(stderr, " -r r     : maximum driver restarts on error, default %d\n", DEFMAXRESTART);
    fprintf(stderr, " -f path  : Path to fifo for dynamic startup and shutdown of drivers.\n");
    fprintf(stderr, " -z       : use zlib compression between INDI servers, default no compression\n");
    fprintf(stderr, " -v       : show key events, no traffic\n");
    fprintf(stderr, " -vv      : -v + key message content\n");
    fprintf(stderr, " -vvv     : -vv + complete xml\n");
    fprintf(stderr, "driver    : executable or [device]@host[:port]\n");

    exit(2);
}

/* arrange for no zombies if drivers die */
//static void noZombies()
//{
//    struct sigaction sa;
//    sa.sa_handler = SIG_IGN;
//    sigemptyset(&sa.sa_mask);
//#ifdef SA_NOCLDWAIT
//    sa.sa_flags = SA_NOCLDWAIT;
//#else
//    sa.sa_flags = 0;
//#endif
//    (void)sigaction(SIGCHLD, &sa, NULL);
//}

/* reap zombies when drivers die, in order to leave SIGCHLD unmodified for subprocesses */
static void zombieRaised(int signum, siginfo_t *sig, void *data)
{
    INDI_UNUSED(data);
    switch (signum)
    {
        case SIGCHLD:
            fprintf(stderr, "Child process %d died\n", sig->si_pid);
            waitpid(sig->si_pid, NULL, WNOHANG);
            break;

        default:
            fprintf(stderr, "Received unexpected signal %d\n", signum);
    }
}

/* reap zombies as they die */
static void reapZombies()
{
    struct sigaction sa;
    sa.sa_sigaction = zombieRaised;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    (void)sigaction(SIGCHLD, &sa, NULL);
}

/* turn off SIGPIPE on bad write so we can handle it inline */
static void noSIGPIPE()
{
    struct sigaction sa;
    sa.sa_handler = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    (void)sigaction(SIGPIPE, &sa, NULL);
}

static DvrInfo *allocDvr()
{
    DvrInfo *dp = NULL;
    int dvi;

    /* try to reuse a driver slot, else add one */
    for (dvi = 0; dvi < ndvrinfo; dvi++)
        if (!(dp = &dvrinfo[dvi])->active)
            break;
    if (dvi == ndvrinfo)
    {
        /* grow dvrinfo */
        dvrinfo = (DvrInfo *)realloc(dvrinfo, (ndvrinfo + 1) * sizeof(DvrInfo));
        if (!dvrinfo)
        {
            fprintf(stderr, "no memory for new drivers\n");
            Bye();
        }
        dp = &dvrinfo[ndvrinfo++];
    }

    if (dp == NULL)
        return NULL;

    /* rig up new dvrinfo entry */
    memset(dp, 0, sizeof(*dp));
    dp->active = 1;
    dp->ndev   = 0;

    return dp;
}

/* open a connection to the given host and port or die.
 * return socket fd.
 */
static int openINDIServer(char host[], int indi_port)
{
    struct sockaddr_in serv_addr;
    struct hostent *hp;
    int sockfd;

    /* lookup host address */
    hp = gethostbyname(host);
    if (!hp)
    {
        fprintf(stderr, "gethostbyname(%s): %s\n", host, strerror(errno));
        Bye();
    }

    /* create a socket to the INDI server */
    (void)memset((char *)&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = ((struct in_addr *)(hp->h_addr_list[0]))->s_addr;
    serv_addr.sin_port        = htons(indi_port);
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        fprintf(stderr, "socket(%s,%d): %s\n", host, indi_port, strerror(errno));
        Bye();
    }

    /* connect */
    errno = 0; /* ensure errno is 0 on success or non-zero on failure */
    if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        fprintf(stderr, "connect(%s,%d): %s\n", host, indi_port, strerror(errno));
        /* Drop thru with valid but unconnected fd, instead of Bye();
         * errno will be non-zero, which will cause trigger calling code
         * to shut down this driver
         */
    }
    else
    {
        /* Make connected socket non-blocking */
        int retGET;
        int retSET;
        retGET = fcntl(sockfd, F_GETFL);
        retSET = fcntl(sockfd, F_SETFL, retGET | O_RDWR | O_NONBLOCK);
        if (retGET < 0 | retSET < 0)
        {
            fprintf(stderr, "%s: %d=connect(...)"
                            "; %d=fcntl(%d, F_GETFL)"
                            "; %d=fcntl(%d, F_SETFL, %d | O_RDWR | O_NONBLOCK)"
                            "; %d=errno[%s]\n"
                          , indi_tstamp(NULL), sockfd
                          , retGET, sockfd
                          , retSET, sockfd, retGET
                          , errno, strerror(errno)
                   );
            /* Same drop thru logic as above (non-zero errno) */
        }
    }

    /* ok */
    return (sockfd);
}

/* return pointer to one new nulled Msg
 */
static Msg *newMsg(void)
{
    return ((Msg *)calloc(1, sizeof(Msg)));
}

/* free Msg mp and everything it contains */
static void freeMsg(Msg *mp)
{
    if (mp->cp && mp->cp != mp->buf)
        free(mp->cp);
    free(mp);
}

/* Shut down read and write ends of an INDI connection
   - If it is a socket, then it is bidirectional and
     the read and write file descriptors will be the same
   - There nay be a gzFile pointers for either or both;
     these will be closed
 */
static void closeINDIconnection(ClInfo* cp, DvrInfo* dp)
{
    int wfd = cp ? cp->s : (dp ? dp->wfd : -1);
    int rfd = cp ? cp->s : (dp ? dp->rfd : -1);
    int wfdvalid = wfd > -1;
    int rfdvalid = rfd > -1;
    int isbidir = rfd == wfd;
    gzFile* pGzfird = cp ? &cp->gzfird : (dp ? &dp->gzfird : NULL);
    gzFile* pGzfiwr = cp ? &cp->gzfiwr : (dp ? &dp->gzfiwr : NULL);

    if (wfdvalid)
    {
        /* Flush any compressed output, if compression is active */
        if (pGzfiwr && *pGzfiwr) { gzflush(*pGzfiwr, Z_FINISH); }

        /* If read and write FDs (file descriptors) are valid and the
         * same, then assume it's a bidirectional socket and shut it
         * down
         */
        if (isbidir) { shutdown(wfd, SHUT_RDWR); }
        /* N.B. that does not close the socket FD; that is done next */
    }

    /* Close write side of INDI connection */
    errno = 0;
    if (wfdvalid)
    {
        if (pGzfiwr && *pGzfiwr)
        {
            gzclose(*pGzfiwr);  // N.B. this will close wfd
            *pGzfiwr = NULL;
        }
        else
        {
            close(wfd);
        }
    }

    errno = 0;
    if (!pGzfird || !*pGzfird) // No read gzFile pointer:  close read FD
    {
        if (isbidir) { return; } // Bidirectional read FD already closed
        if (!rfdvalid) { return; } // No need to close invalid read FD

        close(rfd);  // close valid, non-bidirectional read FD

        return; // the read gzFile pointer is NULL:  nothing more to do
    }

    /* To here the read non-NULL gzFile pointer needs to be closed.
     * If the read FD was already closed by the write logic above, then
     * create an open FD to stand in for it so the gzclose call on the
     * read side will have something to close
     */
    if (isbidir)
    {
        int pair[2];
        int pipertn = pipe(pair);
        dup2(pair[0], rfd);
        close(pair[1]);
    }

    gzclose(*pGzfird);                 /* Close the read gzFile pointer */
    *pGzfird = NULL;
} // static void closeINDIconnection(ClInfo* cp, DvrInfo* dp)

/* close down the given client */
static void shutdownClient(ClInfo *cp)
{
    Msg *mp;

    /* close socket connection */
    shutdown(cp->s, SHUT_RDWR);
    closeINDIconnection(cp, NULL);

    /* free memory */
    delLilXML(cp->lp);
    free(cp->props);

    /* decrement and possibly free any unsent messages for this client */
    while ((mp = (Msg *)popFQ(cp->msgq)) != NULL)
        if (--mp->count == 0)
            freeMsg(mp);
    delFQ(cp->msgq);

    /* ok now to recycle */
    cp->active = 0;

    if (verbose > 0)
        fprintf(stderr, "%s: Client %d: shut down complete - bye!\n", indi_tstamp(NULL), cp->s);
#ifdef OSX_EMBEDED_MODE
    int active = 0;
    for (int i = 0; i < nclinfo; i++)
        if (clinfo[i].active)
            active++;
    fprintf(stderr, "CLIENTS %d\n", active);
    fflush(stderr);
#endif
}

/* write the next chunk of the current message in the queue to the given
 * client. pop message from queue when complete and free the message if we are
 * the last one to use it. shut down this client if trouble.
 * N.B. we assume we will never be called with cp->msgq empty.
 * return 0 if ok else -1 if had to shut down.
 */
static int sendClientMsg(ClInfo *cp)
{
    ssize_t nsend;
    ssize_t nw;
    ssize_t gzwrote;
    Msg *mp;

    /* get current message */
    mp = (Msg *)peekFQ(cp->msgq);

    /* send next chunk, never more than MAXWSIZ to reduce blocking */
    nsend = mp->cl - cp->nsent;
    if (nsend > MAXWSIZ)
        nsend = MAXWSIZ;

    /*******************************************/
    /* Here is the beef:  data are written ... */
    errno = 0;
    if (cp->gzfiwr)
    {
        /* ... compressed, ... */
        gzclearerr(cp->gzfiwr);
        gzwrote = nw = gzwrite(cp->gzfiwr, &mp->cp[cp->nsent], nsend);
        gzflush(cp->gzfiwr, Z_SYNC_FLUSH);
    }
    else
    {
        /* ... or compressed */
        gzwrote = 0;
        nw = write(cp->s, &mp->cp[cp->nsent], nsend);
    }
    /*******************************************/

    /* shut down if trouble */
    if (nw <= 0)
    {
#       if 0
        if (nw == 0)
            fprintf(stderr, "%s: Client %d: write returned 0\n", indi_tstamp(NULL), cp->s);
        else
#       endif
            fprintf(stderr, "%s: Client %d: %swrite returned %ld: errno[%s]; gzerror[%s]\n"
                          , indi_tstamp(NULL), cp->s
                          , cp->gzfiwr ? "gz" : ""
                          , cp->gzfiwr ? gzwrote : nw
                          , strerror(errno)
                          , cp->gzfiwr ? gzerror(cp->gzfiwr,NULL) : "N/A"
                   );
        shutdownClient(cp);
        return (-1);
    }

    /* trace */
    if (verbose > 2)
    {
        char* ts = indi_tstamp(NULL);
        char* ptr = mp->cp + cp->nsent;
        char* ptrend = ptr + nw;
        char* ptrnl;

        fprintf(stderr, "%s: Client %d: sending msg copy %d nq %d:\n"
               , ts, cp->s, mp->count, nFQ(cp->msgq));

        /* Break message string at newlines, so xindiserver does not
         * read a line of logged data from STDERR without a timestamp
         */
        while (ptr < ptrend)
        {
            /* Find first newline in remaining characters
             * N.B. strnchr(...) does not exist
             */
            for (ptrnl=ptr; ptrnl<ptrend && '\n'!=*ptrnl; ++ptrnl) ;
            if (ptr < ptrnl)
            {
              fprintf(stderr, "%s: Client %d: %swrote[%ld] %.*s\n"
                            , ts, cp->s
                            , cp->gzfiwr ? "gz" : ""
                            , cp->gzfiwr ? gzwrote : nw
                            , (int)(ptrnl-ptr), ptr
                     );
            }
            ptr = ptrnl + 1;
        }
    }
    else if (verbose > 1)
    {
        /* Ensure no newline is written with logged data, so xindiserver
         * does not read a line of logged data from STDERR without a
         * timestamp
         */
        char* ptr = mp->cp + cp->nsent;
        char* ptr50 = ptr + 50;
        char* ptrnl;
        for (ptrnl=ptr; ptrnl<ptr50 && *ptrnl!='\n'; ++ptrnl) ;
        fprintf(stderr, "%s: Client %d: sending %.*s\n" , indi_tstamp(NULL), cp->s, (int)(ptrnl-ptr), ptr               );
     /* fprintf(stderr, "%s: Client %d: sending %.50s\n", indi_tstamp(NULL), cp->s                  , &mp->cp[cp->nsent]); */
    }

    /* update amount sent. when complete: free message if we are the last
     * to use it and pop from our queue.
     */
    cp->nsent += nw;
    if (cp->nsent == mp->cl)
    {
        if (--mp->count == 0)
            freeMsg(mp);
        popFQ(cp->msgq);
        cp->nsent = 0;
    }

    return (0);
}

/* return 0 if cp may be interested in dev/name else -1
 */
static int findClDevice(ClInfo *cp, const char *dev, const char *name)
{
    int i;

    if (cp->allprops >= 1 || !dev[0])
        return (0);
    for (i = 0; i < cp->nprops; i++)
    {
        Property *pp = &cp->props[i];
        if (!strcmp(pp->dev, dev) && (!pp->name[0] || !name || !strcmp(pp->name, name)))
            return (0);
    }
    return (-1);
}

/* return size of all Msqs on the given q */
static int msgQSize(FQ *q)
{
    int i, l = 0;

    for (i = 0; i < nFQ(q); i++)
    {
        Msg *mp = (Msg *)peekiFQ(q, i);
        l += sizeof(Msg);
        if (mp->cp != mp->buf)
            l += mp->cl;
    }

    return (l);
}

/* put Msg mp on queue of each client interested in dev/name, except notme.
 * if BLOB always honor current mode.
 * return -1 if had to shut down any clients, else 0.
 */
static int q2Clients(ClInfo *notme, int isblob, const char *dev, const char *name, Msg *mp, XMLEle *root)
{
    int shutany = 0;
    ClInfo *cp;
    int ql, i = 0;

    /* queue message to each interested client */
    for (cp = clinfo; cp < &clinfo[nclinfo]; cp++)
    {
        /* cp in use? notme? want this dev/name? blob? */
        if (!cp->active || cp == notme)
            continue;
        if (findClDevice(cp, dev, name) < 0)
            continue;

        //if ((isblob && cp->blob==B_NEVER) || (!isblob && cp->blob==B_ONLY))
        if (!isblob && cp->blob == B_ONLY)
            continue;

        if (isblob)
        {
            if (cp->nprops > 0)
            {
                Property *pp   = NULL;
                int blob_found = 0;
                for (i = 0; i < cp->nprops; i++)
                {
                    pp = &cp->props[i];
                    if (!strcmp(pp->dev, dev) && (!strcmp(pp->name, name)))
                    {
                        blob_found = 1;
                        break;
                    }
                }

                if ((blob_found && pp->blob == B_NEVER) || (blob_found == 0 && cp->blob == B_NEVER))
                    continue;
            }
            else if (cp->blob == B_NEVER)
                continue;
        }

        /* shut down this client if its q is already too large */
        ql = msgQSize(cp->msgq);
        if (isblob && maxstreamsiz > 0 && ql > maxstreamsiz)
        {
            // Drop frames for streaming blobs
            /* pull out each name/BLOB pair, decode */
            XMLEle *ep      = NULL;
            int streamFound = 0;
            for (ep = nextXMLEle(root, 1); ep; ep = nextXMLEle(root, 0))
            {
                if (strcmp(tagXMLEle(ep), "oneBLOB") == 0)
                {
                    XMLAtt *fa = findXMLAtt(ep, "format");

                    if (fa && strstr(valuXMLAtt(fa), "stream"))
                    {
                        streamFound = 1;
                        break;
                    }
                }
            }
            if (streamFound)
            {
                if (verbose > 1)
                    fprintf(stderr, "%s: Client %d: %d bytes behind. Dropping stream BLOB...\n", indi_tstamp(NULL),
                            cp->s, ql);
                continue;
            }
        }
        if (ql > maxqsiz)
        {
            if (verbose)
                fprintf(stderr, "%s: Client %d: %d bytes behind, shutting down\n", indi_tstamp(NULL), cp->s, ql);
            shutdownClient(cp);
            shutany++;
            continue;
        }

        /* ok: queue message to this client */
        mp->count++;
        pushFQ(cp->msgq, mp);
        if (verbose > 1)
            fprintf(stderr, "%s: Client %d: queuing <%s device='%s' name='%s'>\n", indi_tstamp(NULL), cp->s,
                    tagXMLEle(root), findXMLAttValu(root, "device"), findXMLAttValu(root, "name"));
    }

    return (shutany ? -1 : 0);
}

/* print root as content in Msg mp.
 */
static void setMsgXMLEle(Msg *mp, XMLEle *root)
{
    /* want cl to only count content, but need room for final \0 */
    mp->cl = sprlXMLEle(root, 0);
    if (mp->cl < sizeof(mp->buf))
        mp->cp = mp->buf;
    else
        mp->cp = (char*) malloc(mp->cl + 1);
    sprXMLEle(mp->cp, root, 0);
}

/* save str as content in Msg mp.
 */
static void setMsgStr(Msg *mp, char *str)
{
    /* want cl to only count content, but need room for final \0 */
    mp->cl = strlen(str);
    if (mp->cl < sizeof(mp->buf))
        mp->cp = mp->buf;
    else
        mp->cp = (char*) malloc(mp->cl + 1);
    strcpy(mp->cp, str);
}

static DvrInfo* findActiveDvrInfo(char* name)
{
    if (!name) { return NULL; }
    if (!*name) { return NULL; }
    for (pDvr pdvr = dvrinfo; pdvr<(dvrinfo + ndvrinfo); ++pdvr)
    {
        if (!strcmp(pdvr->name,name) && pdvr->active) { return pdvr; }
    }
    return NULL;
}

ppDvr findDvrInRestartList(pDvr dp)
{
    ppDvr ppdvr = &pRestarts;
    /* Loop over linked list of drivers to restart ... */
    while (*ppdvr)
    {
        /* ... if current driver is in that linked list ... */
        if (*ppdvr == dp) { break; }
        ppdvr = &(*ppdvr)->pNextToRestart;
    }
    return ppdvr;
}

/* Remove driver pointer from linked list of drivers to restart
 */
void removeDvrFromRestartList(pDvr dp)
{
    ppDvr ppdvr = findDvrInRestartList(dp);
    if (dp == *ppdvr)
    {
        *ppdvr = dp->pNextToRestart;
        fprintf(stderr, "%s: Driver %s: removed from restart list.\n"
               , indi_tstamp(NULL), dp->name);
    }
}

/* Add driver pointer to linked list of drivers to restart
 */
void addDvrToRestartList(pDvr dp)
{
    /* Ensure driver is not currently in the list */
    removeDvrFromRestartList(dp);

    /* Add pointer to end of linked list */
    ppDvr ppdvr = &pRestarts;
    while (*ppdvr) { ppdvr = &(*ppdvr)->pNextToRestart; }
    *ppdvr = dp;
    dp->pNextToRestart = NULL;     /* terminate linked list */

    /* Prevent later reuse, set 10s delay until restart */
    dp->active = 1;
    dp->restartDelayus = 10 * Mus;

    fprintf(stderr, "%s: Driver %s: scheduled for restart #%d in %lfs\n"
           , indi_tstamp(NULL), dp->name, ++dp->restarts
           , dp->restartDelayus / ((double)Mus));
}

void handle_restart_list(struct timeval* ptv, void (*startDvr)(pDvr))
{
    /* Calculate time spent in select(2) call from remaining time in tv;
     * ensure time spend is at least 1us
     */
    int time_in_select = ((SELECT_WAITs - ptv->tv_sec) * Mus) - ptv->tv_usec;
    time_in_select = time_in_select > 0 ? time_in_select : 1;

    /* Loop over linked list of drivers to restart */
    pDvr pdvr;
    for (pdvr = pRestarts; pdvr; pdvr = pdvr->pNextToRestart)
    {
        /* Reduce remaining delay by time spent in select(2) call
         * Do nothing more with this driver if delay has not expired
         */
        pdvr->restartDelayus -= time_in_select;
        if (pdvr->restartDelayus > 0) continue;

        /* Drop this driver from the to-be-restarted linked list */
        pRestarts = pdvr->pNextToRestart;

        startDvr(pdvr);
    }
}
static void shutdownDvr(DvrInfo*, int); /* Declare for startRemoteDvr */

/* start the given remote INDI driver connection.
 * exit if trouble.
 */
static void startRemoteDvr(DvrInfo *dp)
{
    Msg *mp;
    char dev[MAXINDIDEVICE] = {0};
    char host[MAXSBUF] = {0};
    char buf[MAXSBUF] = {0};
    int indi_port, sockfd;

    /* extract host and port */
    indi_port = INDIPORT;
    if (sscanf(dp->name, "%[^@]@%[^:]:%d", dev, host, &indi_port) < 2)
    {
        // Device missing? Try a different syntax for all devices
        if (sscanf(dp->name, "@%[^:]:%d", host, &indi_port) < 1)
        {
            fprintf(stderr, "Bad remote device syntax: %s\n", dp->name);
            Bye();
        }
    }

    /* connect */
    sockfd = openINDIServer(host, indi_port);
    int save_errno = errno;

    /* record flag pid, io channels, init lp and snoop list */
    dp->pid = REMOTEDVR;
    strncpy(dp->host, host, MAXSBUF);
    dp->port    = indi_port;
    dp->rfd     = sockfd;
    dp->wfd     = sockfd;
    dp->gzfird  = NULL;
    dp->gzfird  = gzdopen(dp->rfd, "r");
    // If cmd-line had -z, then assume server can do zlib decompression
    dp->gzfiwr  = use_is_zlib ? gzdopen(dp->wfd,"w9") : NULL;
    dp->lp      = newLilXML();
    dp->msgq    = newFQ(1);
    dp->sprops  = (Property *)malloc(1); /* seed for realloc */
    dp->nsprops = 0;
    dp->nsent   = 0;
    dp->active  = 1;
    dp->ndev    = 1;
    dp->dev     = (char **)malloc(sizeof(char *));
    dp->restartDelayus = 0;

    /* N.B. storing name now is key to limiting outbound traffic to this
     * dev.
     */
    dp->dev[0] = (char *)malloc(MAXINDIDEVICE * sizeof(char));
    strncpy(dp->dev[0], dev, MAXINDIDEVICE - 1);
    dp->dev[0][MAXINDIDEVICE - 1] = '\0';

    /* Sending getProperties with device lets remote server limit its
     * outbound (and our inbound) traffic on this socket to this device.
     */
    mp = newMsg();
    pushFQ(dp->msgq, mp);
    if (dev[0])
        sprintf(buf, "<getProperties"
                     " device='%s'"  // device='<dp->dev[0]>' or device='*'
                     "%s"  // message='~~gzready~~' or nothing
                     " version='%g'/>\n"

                   // " device='*'" informs downstream server that it is
                   // connecting to (accepting) an upstream server and
                   // not a regular client. The difference is in how it
                   // treats snooping properties among properties.
                   , dev[0] ? dp->dev[0] : "*"

                   // " message='~~gzready~~'" informs downstream server
                   // that the upstream connection is reading data using
                   // gzread(), so the downstream server is free to use,
                   // or not use, gzwrite() when sending data upstream
                   , dp->gzfird ? " message='~~gzready~~'" : ""

                   , INDIV // version e.g. 1.7 or similar
               );
    setMsgStr(mp, buf);
    mp->count++;

    if (verbose > 0)
        fprintf(stderr, "%s: Driver %s: socket=%d\n", indi_tstamp(NULL), dp->name, sockfd);

    if (save_errno) { shutdownDvr(dp, 1); }  /* oops; try again later */
}

/* start the given local INDI driver process.
 * exit if trouble.
 */
static void startLocalDvr(DvrInfo *dp)
{
    Msg *mp;
    char buf[64];
    int fdstdin;
    int fdstdout;
    /*int fdctrl; */

#ifdef OSX_EMBEDED_MODE
    fprintf(stderr, "STARTING \"%s\"\n", dp->name);
    fflush(stderr);
#endif

    /* build three pipes: r, w and error*/
    if ((fdstdin=open_named_fifo(O_WRONLY, dp->name, ".in", NULL)) < 0)
    {
        fprintf(stderr, "%s: stdin pipe: %s\n", indi_tstamp(NULL), strerror(errno));
        Bye();
    }
    if ((fdstdout=open_named_fifo(O_RDONLY, dp->name, ".out", NULL)) < 0)
    {
        fprintf(stderr, "%s: stdout pipe: %s\n", indi_tstamp(NULL), strerror(errno));
        Bye();
    }
#   if 0
    if ((fdctrl=open_named_fifo(O_RDONLY, dp->name, ".ctrl", NULL)) < 0)
    {
        fprintf(stderr, "%s: stderr pipe: %s\n", indi_tstamp(NULL), strerror(errno));
        Bye();
    }
#   endif/*0*/

    /* record pid, io channels, init lp and snoop list */
    dp->pid = LOCALDVR;
    strncpy(dp->host, "localhost", MAXSBUF);
    dp->port    = -1;
    dp->wfd     = fdstdin;
    dp->rfd     = fdstdout;
    dp->gzfird  = NULL;
    dp->gzfiwr  = NULL;
#   if 0
    dp->efd     = fdctrl;
#   endif/*0*/
    dp->lp      = newLilXML();
    dp->msgq    = newFQ(1);
    dp->sprops  = (Property *)malloc(1); /* seed for realloc */
    dp->nsprops = 0;
    dp->nsent   = 0;
    dp->active  = 1;
    dp->ndev    = 0;
    dp->dev     = (char **)malloc(sizeof(char *));
    dp->restartDelayus = 0;

    /* first message primes driver to report its properties -- dev known
     * if restarting
     */
    mp = newMsg();
    pushFQ(dp->msgq, mp);
    snprintf(buf, sizeof(buf), "<getProperties version='%g'/>\n", INDIV);
    setMsgStr(mp, buf);
    mp->count++;

    if (verbose > 0)
        fprintf(stderr, "%s: Driver %s: pid=%d rfd=%d wfd=%d\n", indi_tstamp(NULL), dp->name, dp->pid, dp->rfd,
                dp->wfd);
}

/* start the given INDI driver process or connection.
 * exit if trouble.
 */
static void startDvr(DvrInfo *dp)
{
    if (strchr(dp->name, '@'))
        startRemoteDvr(dp);
    else
        startLocalDvr(dp);
}

/* close down the given driver and restart */
static void shutdownDvr(DvrInfo *dp, int restart)
{
    Msg *mp;
    int i = 0;

    // Tell any snooping clients that driver is dead.
    for (i = 0; i < dp->ndev; i++)
    {
        /* Inform clients that this driver is dead */
        XMLEle *root = addXMLEle(NULL, (char*)"delProperty");
        addXMLAtt(root, (char*)"device", dp->dev[i]);

        /* Ensure timestamp is prefixed to line sent to STDERR */
        fprintf(stderr, "%s: Driver shutdown: ", indi_tstamp(NULL));
        prXMLEle(stderr, root, 0);
        Msg *mp = newMsg();

        q2Clients(NULL, 0, dp->dev[i], NULL, mp, root);
        if (mp->count > 0)
            setMsgXMLEle(mp, root);
        else
            freeMsg(mp);
        delXMLEle(root);
    }

    /* reclaim resources (connections) */
    closeINDIconnection(NULL, dp);

#ifdef OSX_EMBEDED_MODE
    fprintf(stderr, "STOPPED \"%s\"\n", dp->name);
    fflush(stderr);
#endif

    /* free memory; ensure no double-free if stop interrupts restart */
    if (dp->sprops) { free(dp->sprops); dp->sprops = NULL; }
    if (dp->dev) { free(dp->dev); dp->dev = NULL; }
    if (dp->lp) { delLilXML(dp->lp); dp->lp = NULL; }

    /* ok now to recycle */
    dp->active = 0;
    dp->ndev   = 0;

    /* decrement and possibly free any unsent messages for this client */
    if (dp->msgq)
    {
        /* decrement and possibly free any unsent messages for this client */
        while ((mp = (Msg *)popFQ(dp->msgq)) != NULL)
            if (--mp->count == 0)
                freeMsg(mp);
        delFQ(dp->msgq);
        dp->msgq = NULL;
    }

    if (restart)
    {
        if (dp->restarts >= maxrestarts && maxrestarts > 0)
        {
            fprintf(stderr, "%s: Driver %s: Terminated after #%d restarts.\n", indi_tstamp(NULL), dp->name,
                    dp->restarts);
            // If we're not in FIFO mode and we do not have any more drivers, shutdown the server
            terminateddrv++;
            if ((ndvrinfo - terminateddrv) <= 0 && !fifo.name)
                Bye();
        }
        else
        {
            addDvrToRestartList(dp);
        }
    }
}

/* write the next chunk of the current message in the queue to the given
 * driver. pop message from queue when complete and free the message if we are
 * the last one to use it. restart this driver if touble.
 * N.B. we assume we will never be called with dp->msgq empty.
 * return 0 if ok else -1 if had to shut down.
 */
static int sendDriverMsg(DvrInfo *dp)
{
    ssize_t nsend;
    ssize_t nw;
    ssize_t gzwrote;
    Msg *mp;

    /* get current message */
    mp = (Msg *)peekFQ(dp->msgq);

    /* send next chunk, never more than MAXWSIZ to reduce blocking */
    nsend = mp->cl - dp->nsent;
    if (nsend > MAXWSIZ)
        nsend = MAXWSIZ;

    /*******************************************/
    /* Here is the beef:  data are written ... */
    errno = 0;
    if (dp->gzfiwr)
    {
        /* ... compressed, ... */
        gzclearerr(dp->gzfiwr);
        gzwrote = nw = gzwrite(dp->gzfiwr, &mp->cp[dp->nsent], nsend);
        gzflush(dp->gzfiwr, Z_SYNC_FLUSH);
    }
    else
    {
        /* ... or compressed */
        gzwrote = 0;
        nw = write(dp->wfd, &mp->cp[dp->nsent], nsend);
    }
    /*******************************************/

    /* restart if trouble */
    if (nw <= 0)
    {
#       if 0
        if (nw == 0)
            fprintf(stderr, "%s: Driver %s[wfd=%d]: write returned 0\n", indi_tstamp(NULL), dp->name, dp->wfd);
        else
            fprintf(stderr, "%s: Driver %s[wfd=%d]: write: %s\n", indi_tstamp(NULL), dp->name, dp->wfd, strerror(errno));
#       endif
        fprintf(stderr, "%s: Client %d: %swrite returned %ld: errno[%s]; gzerror[%s]\n"
                       , indi_tstamp(NULL), dp->wfd
                       , dp->gzfiwr ? "gz" : ""
                       , dp->gzfiwr ? gzwrote : nw
                       , strerror(errno)
                       , dp->gzfiwr ? gzerror(dp->gzfiwr,NULL) : "N/A"
                   );
        shutdownDvr(dp, 1);
        return (-1);
    }

    /* trace */
    if (verbose > 2)
    {
        char* ts = indi_tstamp(NULL);
        char* ptr = mp->cp + dp->nsent;
        char* ptrend = ptr + nw;
        char* ptrnl;

        fprintf(stderr, "%s: Driver %s: sending msg copy %d nq %d:\n"
               , ts, dp->name, mp->count, nFQ(dp->msgq));

        /* Break message string at newlines, so xindiserver does not
         * read a line of logged data from STDERR without a timestamp
         */
        while (ptr < ptrend)
        {
            /* Find first newline in remaining characters
             * N.B. strnchr(...) does not exist
             */
            for (ptrnl=ptr; ptrnl<ptrend && '\n'!=*ptrnl; ++ptrnl) ;
            if (ptr < ptrnl)
            {
              fprintf(stderr, "%s: Driver %s: %.*s\n", ts, dp->name, (int)(ptrnl-ptr), ptr);
            }
            ptr = ptrnl + 1;
        }
    }
    else if (verbose > 1)
    {
        /* Ensure no newline is written with logged data, so xindiserver
         * does not read a line of logged data from STDERR without a
         * timestamp
         */
        char* ptr = mp->cp + dp->nsent;
        char* ptr50 = ptr + 50;
        char* ptrnl;
        for (ptrnl=ptr; ptrnl<ptr50 && *ptrnl!='\n'; ++ptrnl) ;
        fprintf(stderr, "%s: Driver %s: sending %.*s\n" , indi_tstamp(NULL), dp->name, (int)(ptrnl-ptr), ptr);
     /* fprintf(stderr, "%s: Driver %s: sending %.50s\n", indi_tstamp(NULL), dp->name                  , &mp->cp[dp->nsent]); */
    }

    /* update amount sent. when complete: free message if we are the last
     * to use it and pop from our queue.
     */
    dp->nsent += nw;
    if (dp->nsent == mp->cl)
    {
        if (--mp->count == 0)
            freeMsg(mp);
        popFQ(dp->msgq);
        dp->nsent = 0;
    }

    return (0);
}

/* create the public INDI Driver endpoint lsocket on port.
 * return server socket else exit.
 */
static void indiListen()
{
    struct sockaddr_in serv_socket;
    int sfd;
    int reuse = 1;

    /* make socket endpoint */
    if ((sfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        fprintf(stderr, "%s: socket: %s\n", indi_tstamp(NULL), strerror(errno));
        Bye();
    }

    /* bind to given port for any IP address */
    memset(&serv_socket, 0, sizeof(serv_socket));
    serv_socket.sin_family = AF_INET;
#ifdef SSH_TUNNEL
    serv_socket.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#else
    serv_socket.sin_addr.s_addr = htonl(INADDR_ANY);
#endif
    serv_socket.sin_port = htons((unsigned short)port);
    if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
    {
        fprintf(stderr, "%s: setsockopt: %s\n", indi_tstamp(NULL), strerror(errno));
        Bye();
    }
    if (bind(sfd, (struct sockaddr *)&serv_socket, sizeof(serv_socket)) < 0)
    {
        fprintf(stderr, "%s: bind: %s\n", indi_tstamp(NULL), strerror(errno));
        Bye();
    }

    /* willing to accept connections with a backlog of 5 pending */
    if (listen(sfd, 5) < 0)
    {
        fprintf(stderr, "%s: listen: %s\n", indi_tstamp(NULL), strerror(errno));
        Bye();
    }

    /* ok */
    lsocket = sfd;
    if (verbose > 0)
        fprintf(stderr, "%s: listening to port %d on fd %d\n", indi_tstamp(NULL), port, sfd);
}

/* Attempt to open up FIFO */
static void indiFIFO(void)
{
    close(fifo.fd);
    fifo.fd = -1;

    /* Open up FIFO, if available */
    if (fifo.name)
    {
        fifo.fd = open(fifo.name, O_RDWR | O_NONBLOCK);

        if (fifo.fd < 0)
        {
            fprintf(stderr, "%s: open(%s): %s.\n", indi_tstamp(NULL), fifo.name, strerror(errno));
            Bye();
        }
    }
}

int isDeviceInDriver(const char *dev, DvrInfo *dp)
{
    int i = 0;
    for (i = 0; i < dp->ndev; i++)
    {
        if (!strcmp(dev, dp->dev[i]))
            return 1;
    }

    return 0;
}

/* Read commands from FIFO and process them. Start/stop drivers accordingly */
static void newFIFO(void)
{
    //char line[MAXRBUF], tDriver[MAXRBUF], tConfig[MAXRBUF], tDev[MAXRBUF], tSkel[MAXRBUF], envDev[MAXRBUF], envConfig[MAXRBUF], envSkel[MAXR];
    char line[MAXRBUF];
    DvrInfo *dp  = NULL;
    int startCmd = 0, i = 0, remoteDriver = 0;
    char* ts = indi_tstamp(NULL);

    while (i < MAXRBUF)
    {
        if (read(fifo.fd, line + i, 1) <= 0)
        {
            // Reset FIFO now, otherwise select will always return with no data from FIFO.
            indiFIFO();
            return;
        }

        if (line[i] == '\n')
        {
            line[i] = '\0';
            i       = 0;
        }
        else
        {
            i++;
            continue;
        }

        if (verbose)
            fprintf(stderr, "%s: FIFO: %s\n", ts, line);

        char cmd[MAXSBUF], arg[4][1], var[4][MAXSBUF], tDriver[MAXSBUF], tName[MAXSBUF], envConfig[MAXSBUF],
             envSkel[MAXSBUF], envPrefix[MAXSBUF];

        memset(&tDriver[0], 0, sizeof(char) * MAXSBUF);
        memset(&tName[0], 0, sizeof(char) * MAXSBUF);
        memset(&envConfig[0], 0, sizeof(char) * MAXSBUF);
        memset(&envSkel[0], 0, sizeof(char) * MAXSBUF);
        memset(&envPrefix[0], 0, sizeof(char) * MAXSBUF);

        int n = 0;

        // If remote driver
        if (strstr(line, "@"))
        {
            n = sscanf(line, "%s %512[^\n]", cmd, tDriver);

            // Remove quotes if any
            char *ptr = tDriver;
            int len   = strlen(tDriver);
            while ((ptr = strstr(tDriver, "\"")))
            {
                memmove(ptr, ptr + 1, --len);
                ptr[len] = '\0';
            }

            //fprintf(stderr, "Remote Driver: %s\n", tDriver);
            remoteDriver = 1;
        }
        // If local driver
        else
        {
            n = sscanf(line, "%s %s -%1c \"%512[^\"]\" -%1c \"%512[^\"]\" -%1c \"%512[^\"]\" -%1c \"%512[^\"]\"", cmd,
                       tDriver, arg[0], var[0], arg[1], var[1], arg[2], var[2], arg[3], var[3]);
            remoteDriver = 0;
        }

        int n_args = (n - 2) / 2;

        int j = 0;
        for (j = 0; j < n_args; j++)
        {
            //fprintf(stderr, "arg[%d]: %c\n", i, arg[j][0]);
            //fprintf(stderr, "var[%d]: %s\n", i, var[j]);

            if (arg[j][0] == 'n')
            {
                strncpy(tName, var[j], MAXSBUF - 1);
                tName[MAXSBUF - 1] = '\0';

                if (verbose)
                    fprintf(stderr, "%s: With name: %s\n", ts, tName);
            }
            else if (arg[j][0] == 'c')
            {
                strncpy(envConfig, var[j], MAXSBUF - 1);
                envConfig[MAXSBUF - 1] = '\0';

                if (verbose)
                    fprintf(stderr, "%s: With config: %s\n", ts, envConfig);
            }
            else if (arg[j][0] == 's')
            {
                strncpy(envSkel, var[j], MAXSBUF - 1);
                envSkel[MAXSBUF - 1] = '\0';

                if (verbose)
                    fprintf(stderr, "%s: With skeketon: %s\n", ts, envSkel);
            }
            else if (arg[j][0] == 'p')
            {
                strncpy(envPrefix, var[j], MAXSBUF - 1);
                envPrefix[MAXSBUF - 1] = '\0';

                if (verbose)
                    fprintf(stderr, "%s: With prefix: %s\n", ts, envPrefix);
            }
        }

        if (!strcmp(cmd, "start"))
            startCmd = 1;
        else
            startCmd = 0;

        if (startCmd)
        {
            if (verbose)
                fprintf(stderr, "%s: FIFO: Starting driver %s\n", ts, tDriver);

            /* If driver is active ... */
            if ((dp=findActiveDvrInfo(tDriver)))
            {
                /* ... and driver is running
		 *     i.e. is not waiting to restart ...
		 */
                if (!*findDvrInRestartList(dp))
                {
                    if (verbose)
                    {
                        fprintf(stderr, "%s: FIFO: Skipping driver %s that is already started\n", ts, tDriver);
                    }
                    /* ... then skip to the next FIFO command, ... */
                    continue;
                }
                /* ... else remove this driver from the restart list */
                removeDvrFromRestartList(dp);
            }

            if (!dp)
            {
                dp = allocDvr();
                strncpy(dp->name, tDriver, MAXINDINAME);
                dp->name[MAXINDINAME-1] = '\0';
            }

            if (remoteDriver == 0)
            {
                //strncpy(dp->dev, tName, MAXINDIDEVICE);
                strncpy(dp->envDev, tName, MAXSBUF);
                strncpy(dp->envConfig, envConfig, MAXSBUF);
                strncpy(dp->envSkel, envSkel, MAXSBUF);
                strncpy(dp->envPrefix, envPrefix, MAXSBUF);
                startDvr(dp);
            }
            else
                startRemoteDvr(dp);
        }
        else  // stop <tDriver>[ -<options>]
        {
            for (dp = dvrinfo; dp < &dvrinfo[ndvrinfo]; dp++)
            {
                fprintf(stderr, "%s: dp->name: %s - tDriver: %s\n", ts, dp->name, tDriver);
                if (!strcmp(dp->name, tDriver) && dp->active == 1)
                {
                    fprintf(stderr, "%s: name: %s - dp->dev[0]: %s\n"
                                  , ts, tName
                                  , dp->ndev ? dp->dev[0] : "<no device name yet>"
                           );

                    /* If device name is given, check against it before shutting down */
                    //if (tName[0] && strcmp(dp->dev[0], tName))
                    if (tName[0] && isDeviceInDriver(tName, dp) == 0)
                        continue;
                    if (verbose)
                        fprintf(stderr, "%s: FIFO: Shutting down driver: %s\n", ts, tDriver);

                    //                    for (i = 0; i < dp->ndev; i++)
                    //                    {
                    //                        /* Inform clients that this driver is dead */
                    //                        XMLEle *root = addXMLEle(NULL, "delProperty");
                    //                        addXMLAtt(root, "device", dp->dev[i]);

                    //                        prXMLEle(stderr, root, 0);
                    //                        Msg *mp = newMsg();

                    //                        q2Clients(NULL, 0, dp->dev[i], NULL, mp, root);
                    //                        if (mp->count > 0)
                    //                            setMsgXMLEle(mp, root);
                    //                        else
                    //                            freeMsg(mp);
                    //                        delXMLEle(root);
                    //                    }

                    if (verbose) { fprintf(stderr, "%s: FIFO: Shutting down driver: %s\n", ts, tDriver); }
                    removeDvrFromRestartList(dp);
                    shutdownDvr(dp, 0);
                    if (verbose) { fprintf(stderr, "%s: FIFO: Driver Shut down complete: %s\n", ts, tDriver); }
                    break;
                }
            }
        }
    }
}

/* block to accept a new client arriving on lsocket.
 * return private nonblocking socket or exit.
 */
static int newClSocket()
{
    struct sockaddr_in cli_socket;
    socklen_t cli_len;
    int cli_fd;
    int retGET = 0;
    int retSET = 0;

    /* get a private non-blocking connection to new client */
    cli_len = sizeof(cli_socket);
    errno = 0;
    cli_fd  = accept(lsocket, (struct sockaddr *)&cli_socket, &cli_len);
    retGET = fcntl(cli_fd, F_GETFL);
    retSET = fcntl(cli_fd, F_SETFL, retGET | O_RDWR | O_NONBLOCK);
    if (cli_fd < 0 || retGET < 0 | retSET < 0)
    {
        fprintf(stderr, "%s: %d=accept(...)"
                        "; %d=fcntl(%d, F_GETFL)"
                        "; %d=fcntl(%d, F_SETFL, %d | O_RDWR | O_NONBLOCK)"
                        "; %d=errno[%s]\n"
                      , indi_tstamp(NULL), cli_fd
                      , retGET, cli_fd
                      , retSET, cli_fd, retGET
                      , errno, strerror(errno)
               );
        Bye();
    }

    /* ok */
    return (cli_fd);
}

/* prepare for new client arriving on lsocket.
 * exit if trouble.
 */
static void newClient()
{
    ClInfo *cp = NULL;
    int s, cli;

    /* assign new socket */
    s = newClSocket();

    /* try to reuse a clinfo slot, else add one */
    for (cli = 0; cli < nclinfo; cli++)
        if (!(cp = &clinfo[cli])->active)
            break;
    if (cli == nclinfo)
    {
        /* grow clinfo */
        clinfo = (ClInfo *)realloc(clinfo, (nclinfo + 1) * sizeof(ClInfo));
        if (!clinfo)
        {
            fprintf(stderr, "no memory for new client\n");
            Bye();
        }
        cp = &clinfo[nclinfo++];
    }

    if (cp == NULL)
        return;

    /* rig up new clinfo entry */
    memset(cp, 0, sizeof(*cp));
    cp->active = 1;
    cp->s      = s;
    cp->gzfird = NULL;
    cp->gzfiwr = NULL;
    // If cmd-line had -z, then trigger check whether client can do zlib
    cp->gzwchk = use_is_zlib;        // Default is 0 => no check
    cp->gzfird = gzdopen(cp->s, "r");
    cp->lp     = newLilXML();
    cp->msgq   = newFQ(1);
    cp->props  = (Property*) malloc(1);
    cp->nsent  = 0;

    if (verbose > 0)
    {
        struct sockaddr_in addr;
        socklen_t len = sizeof(addr);
        getpeername(s, (struct sockaddr *)&addr, &len);
        fprintf(stderr, "%s: Client %d: new arrival from %s:%d"
                        "(zlib decompression is%s available on this end)"
                        " - welcome!\n", indi_tstamp(NULL), cp->s,
                inet_ntoa(addr.sin_addr), ntohs(addr.sin_port)
                , cp->gzfird ? "" : " not"
               );
    }
#ifdef OSX_EMBEDED_MODE
    int active = 0;
    for (int i = 0; i < nclinfo; i++)
        if (clinfo[i].active)
            active++;
    fprintf(stderr, "CLIENTS %d\n", active);
    fflush(stderr);
#endif
}

/* print key attributes and values of the given xml to stderr.
 */
static void traceMsg(XMLEle *root, char* ts)
{
    static const char *prtags[] =
    {
        "defNumber", "oneNumber", "defText", "oneText", "defSwitch", "oneSwitch", "defLight", "oneLight",
    };
    XMLEle *e;
    const char *msg, *perm, *pcd;
    unsigned int i;

    /* print tag header */
    fprintf(stderr, "%s %s %s %s", tagXMLEle(root), findXMLAttValu(root, "device"), findXMLAttValu(root, "name"),
            findXMLAttValu(root, "state"));
    pcd = pcdataXMLEle(root);
    if (pcd[0])
        fprintf(stderr, " %s", pcd);
    perm = findXMLAttValu(root, "perm");
    if (perm[0])
        fprintf(stderr, " %s", perm);
    msg = findXMLAttValu(root, "message");
    if (msg[0])
        fprintf(stderr, " '%s'", msg);

    /* print each array value */
    for (e = nextXMLEle(root, 1); e; e = nextXMLEle(root, 0))
        for (i = 0; i < sizeof(prtags) / sizeof(prtags[0]); i++)
            if (strcmp(prtags[i], tagXMLEle(e)) == 0)
                fprintf(stderr, "\n%s: ...: %10s='%s'", ts, findXMLAttValu(e, "name"), pcdataXMLEle(e));

    fprintf(stderr, "\n");
}

/* add the given device and property to the devs[] list of client if new.
 */
static void addClDevice(ClInfo *cp, const char *dev, const char *name, int isblob)
{
    if (isblob)
    {
        int i = 0;
        for (i = 0; i < cp->nprops; i++)
        {
            Property *pp = &cp->props[i];
            if (!strcmp(pp->dev, dev) && (name == NULL || !strcmp(pp->name, name)))
                return;
        }
    }
    /* no dups */
    else if (!findClDevice(cp, dev, name))
        return;

    /* add */
    cp->props = (Property *)realloc(cp->props, (cp->nprops + 1) * sizeof(Property));
    Property *pp = &cp->props[cp->nprops++];

    /*ip = pp->dev;
    strncpy (ip, dev, MAXINDIDEVICE-1);
    ip[MAXINDIDEVICE-1] = '\0';

    ip = pp->name;
    strncpy (ip, name, MAXINDINAME-1);
        ip[MAXINDINAME-1] = '\0';*/

    strncpy(pp->dev, dev, MAXINDIDEVICE);
    strncpy(pp->name, name, MAXINDINAME);
    pp->blob = B_NEVER;
}

/* convert the string value of enableBLOB to our B_ state value.
 * no change if unrecognized
 */
static void crackBLOB(const char *enableBLOB, BLOBHandling *bp)
{
    if (!strcmp(enableBLOB, "Also"))
        *bp = B_ALSO;
    else if (!strcmp(enableBLOB, "Only"))
        *bp = B_ONLY;
    else if (!strcmp(enableBLOB, "Never"))
        *bp = B_NEVER;
}

/* Update the client property BLOB handling policy */
static void crackBLOBHandling(const char *dev, const char *name, const char *enableBLOB, ClInfo *cp)
{
    int i = 0;

    /* If we have EnableBLOB with property name, we add it to Client device list */
    if (name[0])
        addClDevice(cp, dev, name, 1);
    else
        /* Otherwise, we set the whole client blob handling to what's passed (enableBLOB) */
        crackBLOB(enableBLOB, &cp->blob);

    /* If whole client blob handling policy was updated, we need to pass that also to all children
       and if the request was for a specific property, then we apply the policy to it */
    for (i = 0; i < cp->nprops; i++)
    {
        Property *pp = &cp->props[i];
        if (!name[0])
            crackBLOB(enableBLOB, &pp->blob);
        else if (!strcmp(pp->dev, dev) && (!strcmp(pp->name, name)))
        {
            crackBLOB(enableBLOB, &pp->blob);
            return;
        }
    }
}

/* put Msg mp on queue of each driver responsible for dev, or all drivers
 * if dev not specified.
 */
static void q2RDrivers(const char *dev, Msg *mp, XMLEle *root)
{
    DvrInfo *dp;
    char *roottag = tagXMLEle(root);

    char lastRemoteHost[MAXSBUF];
    int lastRemotePort = -1;

    /* queue message to each interested driver.
     * N.B. don't send generic getProps to more than one remote driver,
     *   otherwise they all fan out and we get multiple responses back.
     */
    for (dp = dvrinfo; dp < &dvrinfo[ndvrinfo]; dp++)
    {
        int isRemote = (dp->pid == REMOTEDVR);

        if (dp->active == 0 || dp->restartDelayus > 0)
            continue;

        /* driver known to not support this dev */
        if (dev[0] && dev[0] != '*' && isDeviceInDriver(dev, dp) == 0)
            continue;

        /* Only send message to each *unique* remote driver at a particular host:port
         * Since it will be propogated to all other devices there */
        if (!dev[0] && isRemote && !strcmp(lastRemoteHost, dp->host) && lastRemotePort == dp->port)
            continue;

        /* JM 2016-10-30: Only send enableBLOB to remote drivers */
        if (isRemote == 0 && !strcmp(roottag, "enableBLOB"))
            continue;

        /* Retain last remote driver data so that we do not send the same info again to a driver
         * residing on the same host:port */
        if (isRemote)
        {
            strncpy(lastRemoteHost, dp->host, MAXSBUF);
            lastRemotePort = dp->port;
        }

        /* ok: queue message to this driver */
        mp->count++;
        pushFQ(dp->msgq, mp);
        if (verbose > 1)
        {
            fprintf(stderr, "%s: Driver %s: queuing responsible for <%s device='%s' name='%s'>\n", indi_tstamp(NULL),
                    dp->name, tagXMLEle(root), findXMLAttValu(root, "device"), findXMLAttValu(root, "name"));
        }
    }
}

/* return Property if dp is snooping dev/name, else NULL.
 */
static Property *findSDevice(DvrInfo *dp, const char *dev, const char *name)
{
    int i;

    for (i = 0; i < dp->nsprops; i++)
    {
        Property *sp = &dp->sprops[i];
        if (!strcmp(sp->dev, dev) && (!sp->name[0] || !strcmp(sp->name, name)))
            return (sp);
    }

    return (NULL);
}

/* put Msg mp on queue of each driver snooping dev/name.
 * if BLOB always honor current mode.
 */
static void q2SDrivers(DvrInfo *me, int isblob, const char *dev, const char *name, Msg *mp, XMLEle *root)
{
    DvrInfo *dp = NULL;

    for (dp = dvrinfo; dp < &dvrinfo[ndvrinfo]; dp++)
    {
        if (dp->active == 0 || dp->restartDelayus > 0)
            continue;

        Property *sp = findSDevice(dp, dev, name);

        /* nothing for dp if not snooping for dev/name or wrong BLOB mode */
        if (!sp)
            continue;
        if ((isblob && sp->blob == B_NEVER) || (!isblob && sp->blob == B_ONLY))
            continue;
        if (me && me->pid == REMOTEDVR && dp->pid == REMOTEDVR)
        {
            // Do not send snoop data to remote drivers at the same host
            // since they will manage their own snoops remotely
            if (!strcmp(me->host, dp->host) && me->port == dp->port)
                continue;
        }

        /* ok: queue message to this device */
        mp->count++;
        pushFQ(dp->msgq, mp);
        if (verbose > 1)
        {
            fprintf(stderr, "%s: Driver %s: queuing snooped <%s device='%s' name='%s'>\n", indi_tstamp(NULL), dp->name,
                    tagXMLEle(root), findXMLAttValu(root, "device"), findXMLAttValu(root, "name"));
        }
    }
}

/* read more from the given client, send to each appropriate driver when see
 * xml closure. also send all newXXX() to all other interested clients.
 * return -1 if had to shut down anything, else 0.
 */
static int readFromClient(ClInfo *cp)
{
    char buf[MAXRBUF];
    int shutany = 0;
    ssize_t i, nr;

    /* read client */
    errno = 0;
    if (cp->gzfird)
    {
        gzclearerr(cp->gzfird);
        nr = gzread(cp->gzfird, buf, sizeof(buf));
    }
    else
    {
        nr = read(cp->s, buf, sizeof(buf));
    }
    if (nr <= 0)
    {
        if (nr < 0)
            fprintf(stderr, "%s: Client %d: read: %s\n"
                          , indi_tstamp(NULL), cp->s, strerror(errno)
                   );
        else if (verbose > 0)
        {
            fprintf(stderr, "%s: Client %d: read EOF[%s]\n"
                          , indi_tstamp(NULL), cp->s, strerror(errno)
                   );
        }
        shutdownClient(cp);
        return (-1);
    }

    /* process XML, sending when find closure */
    for (i = 0; i < nr; i++)
    {
        char err[1024];
        XMLEle *root = readXMLEle(cp->lp, buf[i], err);
        if (root)
        {
            char *roottag    = tagXMLEle(root);
            const char *dev  = findXMLAttValu(root, "device");
            const char *name = findXMLAttValu(root, "name");
            const char *xmsg = findXMLAttValu(root, "message");
            int isblob       = !strcmp(tagXMLEle(root), "setBLOBVector");
            Msg *mp;

            /* Check whether message with ~~gzready~~ was received;
             * if yes, then try to open write compression gzFile pointer
             * N.B. if gzdopen fails, or gzwchk was not greater than
             *      zero because -z was not on command line, then
             *      data written to this client will not be compressed,
             *      but gzread will be able to process them anyway
             *      because the ZLIB/GZ prefix will never be sent, so
             *      the stream will be interpreted as uncompressed data
             */
            if (cp->gzwchk > 0 && !cp->gzfiwr)
            {
                --cp->gzwchk;
                if (strstr(xmsg,"~~gzready~~"))
                {
                    cp->gzfiwr = gzdopen(cp->s, "w9");
                    fprintf(stderr, "%s: Client %d: does zlib decompression"
                                    "; %s in gzdopen()-ing gzFile for writing"
                                    "; %d attempts remaining\n"
                                  , indi_tstamp(NULL), cp->s
                                  , cp->gzfiwr ? "succeeded" : " failed"
                                  , cp->gzwchk
                           );
                }
            }

            if (verbose > 2)
            {
                char *ts = indi_tstamp(NULL);
                fprintf(stderr, "%s: Client %d: read ", ts, cp->s);
                traceMsg(root,ts);
            }
            else if (verbose > 1)
            {
                fprintf(stderr, "%s: Client %d: read <%s device='%s'"
                                " name='%s'>...\n"
                              , indi_tstamp(NULL), cp->s
                              , tagXMLEle(root)
                              , findXMLAttValu(root, "device")
                              , findXMLAttValu(root, "name")
                       );
            }

            /* snag interested properties.
            * N.B. don't open to alldevs if seen specific dev already, else
            *   remote client connections start returning too much.
            */
            if (dev[0])
            {
                // Signature for CHAINED SERVER
                // Not a regular client.
                if (dev[0] == '*' && !cp->nprops)
                    cp->allprops = 2;
                else
                    addClDevice(cp, dev, name, isblob);
            }
            else if (!strcmp(roottag, "getProperties") && !cp->nprops && cp->allprops != 2)
                cp->allprops = 1;

            /* snag enableBLOB -- send to remote drivers too */
            if (!strcmp(roottag, "enableBLOB"))
                crackBLOBHandling(dev, name, pcdataXMLEle(root), cp);

            /* build a new message -- set content iff anyone cares */
            mp = newMsg();

            /* send message to driver(s) responsible for dev */
            q2RDrivers(dev, mp, root);

            /* JM 2016-05-18: Upstream client can be a chained INDI server. If any driver locally is snooping
            * on any remote drivers, we should catch it and forward it to the responsible snooping driver. */
            /* send to snooping drivers. */
            // JM 2016-05-26: Only forward setXXX messages
            if (!strncmp(roottag, "set", 3))
                q2SDrivers(NULL, isblob, dev, name, mp, root);

            /* echo new* commands back to other clients */
            if (!strncmp(roottag, "new", 3))
            {
                if (q2Clients(cp, isblob, dev, name, mp, root) < 0)
                    shutany++;
            }

            /* set message content if anyone cares else forget it */
            if (mp->count > 0)
                setMsgXMLEle(mp, root);
            else
                freeMsg(mp);
            delXMLEle(root);
        }
        else if (err[0])
        {
            char *ts = indi_tstamp(NULL);
            fprintf(stderr, "%s: Client %d: XML error: %s\n", ts, cp->s, err);
            fprintf(stderr, "%s: Client %d: XML read: %.*s\n", ts, cp->s, (int)nr, buf);
            shutdownClient(cp);
            return (-1);
        }
    }

    return (shutany ? -1 : 0);
}

#if 0
/* read more from the given driver stderr, add prefix and send to our stderr.
 * return 0 if ok else -1 if had to restart.
 */
static int stderrFromDriver(DvrInfo *dp)
{
    static char exbuf[MAXRBUF];
    static int nexbuf;
    ssize_t i, nr;

    /* read more */
    nr = read(dp->efd, exbuf + nexbuf, sizeof(exbuf) - nexbuf);
    if (nr <= 0)
    {
        if (nr < 0)
            fprintf(stderr, "%s: Driver %s: stderr %s\n", indi_tstamp(NULL), dp->name, strerror(errno));
        else
            fprintf(stderr, "%s: Driver %s: stderr EOF\n", indi_tstamp(NULL), dp->name);
        shutdownDvr(dp, 1);
        return (-1);
    }
    nexbuf += nr;

    /* prefix each whole line to our stderr, save extra for next time */
    for (i = 0; i < nexbuf; i++)
    {
        if (exbuf[i] == '\n')
        {
            fprintf(stderr, "%s: Driver %s: %.*s\n", indi_tstamp(NULL), dp->name, (int)i, exbuf);
            i++;                               /* count including nl */
            nexbuf -= i;                       /* remove from nexbuf */
            memmove(exbuf, exbuf + i, nexbuf); /* slide remaining to front */
            i = -1;                            /* restart for loop scan */
        }
    }

    return (0);
}
#endif/*0*/

/* add dev/name to dp's snooping list.
 * init with blob mode set to B_NEVER.
 */
static void addSDevice(DvrInfo *dp, const char *dev, const char *name)
{
    Property *sp;
    char *ip;

    /* no dups */
    sp = findSDevice(dp, dev, name);
    if (sp)
        return;

    /* add dev to sdevs list */
    dp->sprops = (Property *)realloc(dp->sprops, (dp->nsprops + 1) * sizeof(Property));
    sp         = &dp->sprops[dp->nsprops++];

    ip = sp->dev;
    strncpy(ip, dev, MAXINDIDEVICE - 1);
    ip[MAXINDIDEVICE - 1] = '\0';

    ip = sp->name;
    strncpy(ip, name, MAXINDINAME - 1);
    ip[MAXINDINAME - 1] = '\0';

    sp->blob = B_NEVER;

    if (verbose)
        fprintf(stderr, "%s: Driver %s: snooping on %s.%s\n", indi_tstamp(NULL), dp->name, dev, name);
}

/* put Msg mp on queue of each chained server client, except notme.
  * return -1 if had to shut down any clients, else 0.
 */
static int q2Servers(DvrInfo *me, Msg *mp, XMLEle *root)
{
    int shutany = 0, i = 0, devFound = 0;
    ClInfo *cp;
    int ql = 0;

    /* queue message to each interested client */
    for (cp = clinfo; cp < &clinfo[nclinfo]; cp++)
    {
        /* cp in use? not chained server? */
        if (!cp->active)
            continue;

	devFound = 0;

        // Only send the message to the upstream server that is connected specfically to the device in driver dp
        switch (cp->allprops)
        {
            // 0 --> not all props are requested. Check for specific combination
            case 0:
                for (i = 0; i < cp->nprops; i++)
                {
                    Property *pp = &cp->props[i];
                    int j        = 0;
                    for (j = 0; j < me->ndev; j++)
                    {
                        if (!strcmp(pp->dev, me->dev[j]))
                            break;
                    }

                    if (j != me->ndev)
                    {
                        devFound = 1;
                        break;
                    }
                }
            break;

            // All props are requested. This is client-only mode (not upstream server)
            case 1:
                break;
            // Upstream server mode
            case 2:
                devFound = 1;
                break;
        }

        // If no matching device found, continue
        if (devFound == 0)
            continue;

        /* shut down this client if its q is already too large */
        ql = msgQSize(cp->msgq);
        if (ql > maxqsiz)
        {
            if (verbose)
                fprintf(stderr, "%s: Client %d: %d bytes behind, shutting down\n", indi_tstamp(NULL), cp->s, ql);
            shutdownClient(cp);
            shutany++;
            continue;
        }

        /* ok: queue message to this client */
        mp->count++;
        pushFQ(cp->msgq, mp);
        if (verbose > 1)
            fprintf(stderr, "%s: Client %d: queuing <%s device='%s' name='%s'>\n", indi_tstamp(NULL), cp->s,
                    tagXMLEle(root), findXMLAttValu(root, "device"), findXMLAttValu(root, "name"));
    }

    return (shutany ? -1 : 0);
}

/* log message in root known to be from device dev to ldir, if any.
 */
static void logDMsg(XMLEle *root, const char *dev)
{
    char stamp[64];
    char logfn[1024];
    const char *ts, *ms;
    FILE *fp;

    /* get message, if any */
    ms = findXMLAttValu(root, "message");
    if (!ms[0])
        return;

    /* get timestamp now if not provided */
    ts = findXMLAttValu(root, "timestamp");
    if (!ts[0])
    {
        indi_tstamp(stamp);
        ts = stamp;
    }

    /* append to log file, name is date portion of time stamp */
    sprintf(logfn, "%s/%.10s.islog", ldir, ts);
    fp = fopen(logfn, "a");
    if (!fp)
        return; /* oh well */
    fprintf(fp, "%s: %s: %s\n", ts, dev, ms);
    fclose(fp);
}

/* read more from the given driver, send to each interested client when see
 * xml closure. if driver dies, try restarting.
 * return 0 if ok else -1 if had to shut down anything.
 */
static int readFromDriver(DvrInfo *dp)
{
    char buf[MAXRBUF];
    int shutany = 0;
    ssize_t nr;
    char err[1024];
    XMLEle **nodes;
    XMLEle *root;
    int inode = 0;

    /* read driver */
    errno = 0;
    if (dp->gzfird)
    {
        gzclearerr(dp->gzfird);
        nr = gzread(dp->gzfird, buf, sizeof(buf));
    }
    else
    {
        nr = read(dp->rfd, buf, sizeof(buf));
    }
    if (nr <= 0)
    {
        if (nr < 0)
            fprintf(stderr, "%s: Driver %s: stdin %s\n", indi_tstamp(NULL), dp->name, strerror(errno));
        else
            fprintf(stderr, "%s: Driver %s: stdin EOF\n", indi_tstamp(NULL), dp->name);

        shutdownDvr(dp, 1);
        return (-1);
    }

    /* process XML chunk */
    nodes = parseXMLChunk(dp->lp, buf, nr, err);

    if (!nodes)
    {
        if (err[0])
        {
            char *ts = indi_tstamp(NULL);
            fprintf(stderr, "%s: Driver %s: XML error: %s\n", ts, dp->name, err);
            fprintf(stderr, "%s: Driver %s: XML read: %.*s\n", ts, dp->name, (int)nr, buf);
            shutdownDvr(dp, 1);
            return (-1);
        }
        return -1;
    }

    root = nodes[inode];
    while (root)
    {
        char *roottag    = tagXMLEle(root);
        const char *dev  = findXMLAttValu(root, "device");
        const char *name = findXMLAttValu(root, "name");
        int isblob       = !strcmp(tagXMLEle(root), "setBLOBVector");
        Msg *mp;

        if (verbose > 2)
        {
            char *ts = indi_tstamp(NULL);
            fprintf(stderr, "%s: Driver %s: read ", ts, dp->name);
            traceMsg(root,ts);
        }
        else if (verbose > 1)
        {
            fprintf(stderr, "%s: Driver %s: read <%s device='%s' name='%s'>\n", indi_tstamp(NULL), dp->name,
                    tagXMLEle(root), findXMLAttValu(root, "device"), findXMLAttValu(root, "name"));
        }

        /* that's all if driver is just registering a snoop */
        /* JM 2016-05-18: Send getProperties to upstream chained servers as well.*/
        if (!strcmp(roottag, "getProperties"))
        {
            addSDevice(dp, dev, name);
            mp = newMsg();
            /* send to interested chained servers upstream */
            if (q2Servers(dp, mp, root) < 0)
                shutany++;
            /* Send to snooped drivers if they exist so that they can echo back the snooped propertly immediately */
            q2RDrivers(dev, mp, root);

            if (mp->count > 0)
                setMsgXMLEle(mp, root);
            else
                freeMsg(mp);
            delXMLEle(root);
            inode++;
            root = nodes[inode];
            continue;
        }

        /* that's all if driver desires to snoop BLOBs from other drivers */
        if (!strcmp(roottag, "enableBLOB"))
        {
            Property *sp = findSDevice(dp, dev, name);
            if (sp)
                crackBLOB(pcdataXMLEle(root), &sp->blob);
            delXMLEle(root);
            inode++;
            root = nodes[inode];
            continue;
        }

        /* Found a new device? Let's add it to driver info */
        if (dev[0] && isDeviceInDriver(dev, dp) == 0)
        {
            dp->dev           = (char **)realloc(dp->dev, (dp->ndev + 1) * sizeof(char *));
            dp->dev[dp->ndev] = (char *)malloc(MAXINDIDEVICE * sizeof(char));

            strncpy(dp->dev[dp->ndev], dev, MAXINDIDEVICE - 1);
            dp->dev[dp->ndev][MAXINDIDEVICE - 1] = '\0';

#ifdef OSX_EMBEDED_MODE
            if (!dp->ndev)
                fprintf(stderr, "STARTED \"%s\"\n", dp->name);
            fflush(stderr);
#endif

            dp->ndev++;
        }

        /* log messages if any and wanted */
        if (ldir)
            logDMsg(root, dev);

        /* build a new message -- set content iff anyone cares */
        mp = newMsg();

        /* send to interested clients */
        if (q2Clients(NULL, isblob, dev, name, mp, root) < 0)
            shutany++;

        /* send to snooping drivers */
        q2SDrivers(dp, isblob, dev, name, mp, root);

        /* set message content if anyone cares else forget it */
        if (mp->count > 0)
            setMsgXMLEle(mp, root);
        else
            freeMsg(mp);
        delXMLEle(root);
        inode++;
        root = nodes[inode];
    }

    free(nodes);

    return (shutany ? -1 : 0);
}

#ifndef __INDISERVER_MAIN__
#define __INDISERVER_MAIN__ main
#define __CALL_INDIRUN_IN_MAIN__
static
#endif//__INDISERVER_MAIN__
/* service traffic from clients and drivers */
void indiRun(void)
{
    fd_set rs, ws;
    int maxfd = 0;
    int i, s;

    /* init with no writers or readers */
    FD_ZERO(&ws);
    FD_ZERO(&rs);

    if (fifo.name && fifo.fd >= 0)
    {
        FD_SET(fifo.fd, &rs);
        maxfd = fifo.fd;
    }

    /* always listen for new clients */
    FD_SET(lsocket, &rs);
    if (lsocket > maxfd)
        maxfd = lsocket;

    /* add all client readers and client writers with work to send */
    for (i = 0; i < nclinfo; i++)
    {
        ClInfo *cp = &clinfo[i];
        if (cp->active)
        {
            FD_SET(cp->s, &rs);
            if (nFQ(cp->msgq) > 0)
                FD_SET(cp->s, &ws);
            if (cp->s > maxfd)
                maxfd = cp->s;
        }
    }

    /* add all driver readers and driver writers with work to send */
    for (i = 0; i < ndvrinfo; i++)
    {
        DvrInfo *dp = &dvrinfo[i];
        if (dp->active && dp->restartDelayus < 1)
        {
            FD_SET(dp->rfd, &rs);
            if (dp->rfd > maxfd)
                maxfd = dp->rfd;
#           if 0
            if (dp->pid != REMOTEDVR)
            {
                FD_SET(dp->efd, &rs);
                if (dp->efd > maxfd)
                    maxfd = dp->efd;
            }
#           endif/*0*/
            if (nFQ(dp->msgq) > 0)
            {
                FD_SET(dp->wfd, &ws);
                if (dp->wfd > maxfd)
                    maxfd = dp->wfd;
            }
        }
    }

    struct timeval tv = {SELECT_WAITs, 0}; /* {seconds, microseconds} */

    /* wait for action */
    s = select(maxfd + 1, &rs, &ws, NULL, &tv);
    if (s < 0)
    {
        if(errno == EINTR)
            return;
        fprintf(stderr, "%s: select(%d): %s\n", indi_tstamp(NULL), maxfd + 1, strerror(errno));
        Bye();
    }

    /* new command from FIFO? */
    if (s > 0 && fifo.fd >= 0 && FD_ISSET(fifo.fd, &rs))
    {
        newFIFO();
        s--;
    }

    /* new client? */
    if (s > 0 && FD_ISSET(lsocket, &rs))
    {
        newClient();
        s--;
    }

    /* message to/from client? */
    for (i = 0; s > 0 && i < nclinfo; i++)
    {
        ClInfo *cp = &clinfo[i];
        if (cp->active)
        {
            if (FD_ISSET(cp->s, &rs))
            {
                if (readFromClient(cp) < 0)
                    return; /* fds effected */
                s--;
            }
            if (s > 0 && FD_ISSET(cp->s, &ws))
            {
                if (sendClientMsg(cp) < 0)
                    return; /* fds effected */
                s--;
            }
        }
    }

    /* message to/from driver? */
    for (i = 0; s > 0 && i < ndvrinfo; i++)
    {
        DvrInfo *dp = &dvrinfo[i];
        if (dp->active && dp->restartDelayus < 1)
        {
#           if 0
            if (dp->pid != REMOTEDVR && FD_ISSET(dp->efd, &rs))
            {
                if (stderrFromDriver(dp) < 0)
                    return; /* fds effected */
                s--;
            }
#           endif/*0*/
            if (s > 0 && FD_ISSET(dp->rfd, &rs))
            {
                if (readFromDriver(dp) < 0)
                    break; //return; /* fds effected */
                s--;
            }
            if (s > 0 && FD_ISSET(dp->wfd, &ws) && nFQ(dp->msgq) > 0)
            {
                if (sendDriverMsg(dp) < 0)
                    break; //return; /* fds effected */
                s--;
            }
        }
    }

    /* Returns above are now breaks so the restart list is processed */
    handle_restart_list(&tv, startDvr);
}

int __INDISERVER_MAIN__(int ac, char *av[])
{
    /* log startup */
    logStartup(ac, av);

    /* save our name */
    me = av[0];

#ifdef OSX_EMBEDED_MODE

    char logname[128];
    snprintf(logname, 128, LOGNAME, getlogin());
    fprintf(stderr, "switching stderr to %s", logname);
    freopen(logname, "w", stderr);

    fifo.name = FIFONAME;
    verbose   = 1;
    ac        = 0;

#else

    /* crack args */
    while ((--ac > 0) && ((*++av)[0] == '-'))
    {
        char *s;
        for (s = av[0] + 1; *s != '\0'; s++)
            switch (*s)
            {
                case 'l':
                    if (ac < 2)
                    {
                        fprintf(stderr, "-l requires log directory\n");
                        usage();
                    }
                    ldir = *++av;
                    ac--;
                    break;
                case 'm':
                    if (ac < 2)
                    {
                        fprintf(stderr, "-m requires max MB behind\n");
                        usage();
                    }
                    maxqsiz = 1024 * 1024 * atoi(*++av);
                    ac--;
                    break;
                case 'p':
                    if (ac < 2)
                    {
                        fprintf(stderr, "-p requires port value\n");
                        usage();
                    }
                    port = atoi(*++av);
                    ac--;
                    break;
                case 'd':
                    if (ac < 2)
                    {
                        fprintf(stderr, "-d requires max stream MB behind\n");
                        usage();
                    }
                    maxstreamsiz = 1024 * 1024 * atoi(*++av);
                    ac--;
                    break;
                case 'f':
                    if (ac < 2)
                    {
                        fprintf(stderr, "-f requires fifo node\n");
                        usage();
                    }
                    fifo.name = *++av;
                    ac--;
                    break;
                case 'r':
                    if (ac < 2)
                    {
                        fprintf(stderr, "-r requires number of restarts\n");
                        usage();
                    }
                    maxrestarts = atoi(*++av);
                    if (maxrestarts < 0)
                        maxrestarts = 0;
                    ac--;
                    break;
                case 'v':
                    verbose++;
                    break;
                case 'z':
                    use_is_zlib = 1;
                    break;
                default:
                    usage();
            }
    }
#endif

    /* at this point there are ac args in av[] to name our drivers */
    if (ac == 0 && !fifo.name)
        usage();

    /* take care of some unixisms */
    /*noZombies();*/
    reapZombies();
    noSIGPIPE();

    /* realloc seed for client pool */
    clinfo  = (ClInfo *)malloc(1);
    nclinfo = 0;

    /* create driver info array all at once since size never changes */
    ndvrinfo = ac;
    dvrinfo  = (DvrInfo *)calloc(ndvrinfo, sizeof(DvrInfo));

    /* Ensure link list of drivers to start is empty */
    pRestarts = NULL;

    /* start each driver */
    while (ac-- > 0)
    {
        strncpy(dvrinfo[ac].name, *av++, MAXINDINAME);
        startDvr(&dvrinfo[ac]);
    }

    /* announce we are online */
    indiListen();

    /* Load up FIFO, if available */
    indiFIFO();

#ifdef __CALL_INDIRUN_IN_MAIN__
    /* handle new clients and all io */
    while (1)
        indiRun();

    /* whoa! */
    fprintf(stderr, "unexpected return from main\n");
#endif//__CALL_INDIRUN_IN_MAIN__
    return (1);
}
