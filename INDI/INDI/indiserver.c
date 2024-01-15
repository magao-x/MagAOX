/* Copyright (C) 2013 Elwood C. Downey ecdowney@clearskyinstitute.com
 * licensed under GNU Lesser Public License version 2.1.
 *
 * Indiserver is basically a message router among Drivers and Clients. Message
 *   headers are sniffed so messages are only sent to parties that have shown
 *   interest in a given Device. All newXXX() received from one Client are echoed
 *   to all other Clients who have shown an interest in the same Device and property.
 *   This allows all clients for a given Device to remain in sync if they wish. All
 *   messages must conform to INDI protocol defined at
 *   http://www.clearskyinstitute.com/INDI/INDI.pdf.
 * argv lists names of local Driver programs to run or sockets to connect for remote
 *   Devices. Run with --help argument for usage synopsis.
 * Local Drivers are restarted if they exit or their connection closes. Connection
 *   to remote Devices is retried if connection is lost.
 * Each local Driver's stdin/out are assumed to provide INDI traffic and are
 *   connected here via pipes. Local Drivers' stderr are connected to our
 *   stderr with date stamp and driver name prepended. In turn, our stderr
 *   can be connected to the log directory using -l.
 * We only support Drivers that advertise support for one Device. The problem
 *   with multiple Devices in one Driver is without a way to know what they
 *   _all_ are there is no way to avoid sending all messages to all Drivers.
 *   This would also prevent detection and prevention of chained loops.
 *
 * Implementation notes:
 *
 * The main thread starts each Driver then listens for new clients. New clients each get
 * two threads, one for reading and one for writing. Drivers each get three threads,
 * one for reading its stdout, one for reading its stderr, and one for writing to its
 * stdin. Readers distribute new messages onto the queues of the interested writers then
 * signal a condition variable kept by each writer. Writer threads wait on a condition
 * variable for new entries on their message queue. Writers are also notified of problems
 * seen by their corresponding Readers (typically EOF) using the same condition variable.
 * All threads are run detached so never need to be joined.
 *
 * Since one message might be destined to more than one Client or Device, they contain
 * a usage count that is incremented as they are queued for transmission and decremented
 * as they are successfully sent. A message is freed after the last user is finished.
 * Messages are saved in their original XML text form for retransmission, they are not
 * copied or reformated from the parsed XML. Clients or drivers that get more
 * than maxqsiz bytes behind are forcibly shut down.
 *
 * Mutexes:
 *  [] The overall list of clients is guarded by a rwlock as clients come and go.
 *  [] Each client structure contains a mutex to guard its queue of messages.
 *  [] Each client structure contains a rwlock to guard its list of props and blobs.
 *  [] Each driver structure contains a mutex to guard its queue of messages.
 *  [] Each driver structure contains a rwlock to guard its list of snooping devices.
 *  [] Each driver structure contains a rwlock write-locked when/if it is restarted.
 *  [] Each message contains a mutex to guard its usage count.
 *  [] The log file is marshalled by a mutex.
 *
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <ctype.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <arpa/inet.h>

#include "lilxml.h"
#include "indiapi.h"
#include "fq.h"

#define INDIPORT        7624            /* default TCP/IP port to listen */
#define	REMOTEDVR	(-1234)		/* invalid PID to flag remote drivers */
#define	MAXRBUF		40960		/* max read buffer */
#define	MAXWSIZ		40960		/* max bytes/write */
#define	DEFMAXQSIZ	50		/* default max q behind, MB */
#define	RDRTIME		2		/* remote driver retry delay, secs */
#define EXITEXFAIL	98		/* driver execlp failed */
#define	RESTARTDT	10		/* don't restart a driver sooner than this, seconds */
static char lockout_fn[] = "/tmp/noindi";	/* do not restart local driver if this exists */

/* associate a usage count with a single message queued to potentially multiple
 * drivers or clients.
 */
typedef struct {
    pthread_mutex_t count_lock;		/* lock whenever changing count */
    int count;				/* number of consumers left */
    int total;				/* total space at cp[] */
    int used;				/* cp[] space actually in use */
    int next;				/* processing index into cp[] */
    char *cp;				/* content: buf at first then malloced for more */
    char buf[MAXRBUF];			/* local fast buf for most messages */
} Msg;

/* BLOB handling, NEVER is the default */
typedef enum {B_NEVER=0, B_ALSO, B_ONLY} BLOBHandling;

/* device + property name */
typedef struct {
    char dev[MAXINDIDEVICE];
    char name[MAXINDINAME];
} Property;

/* record of each snooped property */
typedef struct {
    Property prop;
    BLOBHandling blob;			/* when to snoop BLOBs */
} Snoopee;

/* info for each connected client.
 * clinfo is a list of pointers to ClInfo pointers so they don't move when list grows.
 * list never shrinks, but entries are reused via active.
 */
typedef struct {
    int active;				/* 1 when this record is in use */
    Property *props;			/* malloced array of props we want */
    int nprops;				/* n entries in props[] */
    Property *blobs;			/* malloced array of BLOBs we want */
    int nblobs;				/* n entries in blobs[] */
    pthread_rwlock_t props_rwlock;	/* guard changes to props and blobs */
    int s;				/* socket for this client */
    struct sockaddr_in addr;		/* client address */
    char addrname[32];			/* client host in ascii */
    int err;				/* set on fatal error */
    LilXML *lp;				/* XML parsing context */
    Msg *mp;				/* new incoming message */
    FQ *msgq;				/* outbound Msg queue  -- guard with q_lock */
    pthread_cond_t go_cond;		/* tell writer thread to send next msqq */
    pthread_mutex_t q_lock;		/* guard access to msqg and go_cond */
} ClInfo;
static ClInfo **clinfo;			/* malloced pool of ptrs to malloced ClInfos */
static int nclinfo;			/* n entries in clinfo */
static pthread_rwlock_t cl_rwlock;	/* guard scanning clinfo */

/* info for each connected driver.
 * list never changes or moves so it can be an array,
 * but some may be locked if/when restarting
 */
typedef struct {
    char *name;				/* programm name or remote description */
    char dev[MAXINDIDEVICE];		/* device served by this driver */
    Snoopee **sprops;			/* malloced array of ptrs to malloced props we snoop */
    int nsprops;			/* n entries in sprops[] */
    pthread_rwlock_t sprops_rwlock;	/* guard changes to sprops */
    int pid;				/* process id or REMOTEDVR flag if remote */
    struct sockaddr_in addr;		/* remote addr, iff (pid == REMOTEDVR) */
    char addrname[32];			/* remote addr in ascii, iff (pid == REMOTEDVR) */
    int err;				/* set on fatal error */
    int rfd;				/* driver's stdout read pipe fd if local, else socket */
    int wfd;				/* driver's stdin write pipe fd if local, else socket  */
    int efd;				/* driver's stderr read pipe fd, if local */
    pthread_t stderr_thr;		/* stderr reader thread */
    time_t start;			/* time this driver was started */
    int restarts;			/* n times this process has been restarted */
    LilXML *lp;				/* XML parsing context */
    Msg *mp;				/* new incoming message */
    FQ *msgq;				/* outbound Msg queue  -- guard with q_lock */
    pthread_cond_t go_cond;		/* tell writer thread to send next msqq */
    pthread_mutex_t q_lock;		/* guard access to msqg and go_cond */
    pthread_rwlock_t restart_lock;	/* lock out this device while restarting */
} DvrInfo;
static DvrInfo *dvrinfo;		/* malloced array of DvrInfo */
static int ndvrinfo;			/* n total */

/* local variables */
static char *me;			/* our argv[0] name */
static int port = INDIPORT;		/* public INDI port */
static int verbose;			/* chattiness, cumulative */
static int profile_exit;		/* exit after last client, just for -x debug  */
static int lsocket;			/* master listen() socket */
static char *ldir;			/* log directory f -l */
static pthread_mutex_t log_lock;	/* lock when writing to our error log */
static int maxqsiz = (DEFMAXQSIZ*1024*1024); /* kill if these many bytes behind */
static int ignore_lockout;              /* whether to honor lockout_fn */

/* local prototypes */
static void logDrivers (int ac, char *av[]);
static void usage (void);
static void allowCoreDumps (void);
static void noSIGPIPE (void);
static void indiListen (void);
static void newClient (void);
static int newClSocket (void);
static void shutdownClient (ClInfo *cp);
static void initDvr (DvrInfo *dp, char *name);
static void startDvr (DvrInfo *dp);
static void *startDvrThread (void *dp);
static void startLocalDvr (DvrInfo *dp);
static void startRemoteDvr (DvrInfo *dp);
static int openRemoteConnection (char host[], int port);
static void restartDvr (DvrInfo *dp);
static void q2Drivers (char *dev, Msg *mp, char *roottag);
static void q2SnoopingDrivers (int isblob, char *dev, char *name, Msg *mp);
static void q2Clients (ClInfo *notme, int isblob, char *dev, char *name, Msg *mp);
static void addSnoopDevice (DvrInfo *dp, char *dev, char *name);;
static Snoopee *findSnoopDevice (DvrInfo *dp, char *dev, char *name);
static void addClDevice (ClInfo *cp, int isblob, char *dev, char *name);
static void rmClDevice (ClInfo *cp, int isblob, char *dev, char *name);
static int findClDevice (ClInfo *cp, int isblob, char *dev, char *name);
static void logMsg (const char *label, DvrInfo *dp, ClInfo *cp, Msg *mp);
static void *driverStdoutReaderThread (void *);
static void *driverStderrReaderThread (void *);
static void *driverWriterThread (void *);
static void *clientReaderThread (void *);
static void *clientWriterThread (void *);
static void onDriverError (DvrInfo *dp);
static void onClientError (ClInfo *cp);
static int pushMsg (DvrInfo *dp, ClInfo *cp, Msg *mp);
static int msgQSize (FQ *q);
static void decMsg (Msg *mp);
static void minMsg (Msg *mp, int add);
static Msg *splitMsg (Msg *mp, int keep);
static Msg *newMsg (void);
static void addMsg (Msg *mp, char buf[], int bufl);
static void incMsg (Msg *mp);
static void drainMsgs (FQ *qp);
static void crackBLOB (char *enableBLOB, BLOBHandling *bp);
static void traceMsg (XMLEle *root);
static char *tstamp (char *s);
static void logDvrMsg (XMLEle *root, char *dev);
static void logMessage (const char *fmt, ...);
static char *strncpyz (char *dst, const char *src, int n);
static void ssleep (int ms);
static void Bye(const char *fmt, ...);

int
main (int ac, char *av[])
{
	/* save our name */
	me = av[0];

	/* crack args */
	while ((--ac > 0) && ((*++av)[0] == '-')) {
	    char *s;
	    for (s = av[0]+1; *s != '\0'; s++)
		switch (*s) {
		case 'l':
		    if (ac < 2) {
			fprintf (stderr, "-l requires log directory\n");
			usage();
		    }
		    ldir = *++av;
		    ac--;
		    break;
		case 'm':
		    if (ac < 2) {
			fprintf (stderr, "-m requires max MB behind\n");
			usage();
		    }
		    maxqsiz = 1024*1024*atoi(*++av);
		    ac--;
		    break;
                case 'n':
                    ignore_lockout++;
                    break;
		case 'p':
		    if (ac < 2) {
			fprintf (stderr, "-p requires port value\n");
			usage();
		    }
		    port = atoi(*++av);
		    ac--;
		    break;
		case 'v':
		    verbose++;
		    break;
		case 'x':
		    profile_exit++;
		    break;
		default:
		    fprintf (stderr, "Unknown option: %c\n", *s);
		    usage();
		}
	}

	/* at this point there are ac args in av[] to name our drivers */
	if (ac == 0) {
	    fprintf (stderr, "Must give at least one driver\n");
	    usage();
	}

	/* prepare log file lock before first use of logMessage() */
	pthread_mutex_init (&log_lock, NULL);

	/* log our list of drivers */
	logDrivers  (ac, av);

	/* take care of some unixisms */
	allowCoreDumps();
	noSIGPIPE();
	close (0);

	/* seed realloc for client pool and prep lock */
	clinfo = (ClInfo **) malloc (1);
	nclinfo = 0;
	pthread_rwlock_init (&cl_rwlock, NULL);

	/* announce we are online before starting remote drivers */
	indiListen();

	/* start each driver */
	ndvrinfo = ac;
	dvrinfo = (DvrInfo *) calloc (ndvrinfo, sizeof(DvrInfo));
	while (ac-- > 0)
	    initDvr (&dvrinfo[ac], *av++);

	/* handle new clients forever */
	while (1)
	    newClient();

	/* whoa! */
	logMessage ("unexpected return from main()\n");
	return (1);
}

/* record we have started and our drivers */
static void
logDrivers (int ac, char *av[])
{
	char buf[32768]; //Just make huge.  2048 was too small.
	int i, l;

	l = snprintf (buf,sizeof(buf), "startup: ");
	for (i = 0; i < ac; i++)
	    l += snprintf (buf+l, sizeof(buf), "%s ", av[i]);
	logMessage ("%s\n", buf);
}

/* print usage message and exit (2) */
static void
usage(void)
{
	fprintf (stderr,"Usage: %s [options] driver [driver ...]\n", me);
	fprintf (stderr,"Purpose: server for local and remote INDI drivers\n");
	fprintf (stderr,"Code %s. Protocol %g.\n", "$Revision: 1.18 $", INDIV);
	fprintf (stderr,"Options:\n");
	fprintf (stderr," -l d  : log messages to <d>/YYYY-MM-DD.islog, else stderr\n");
	fprintf (stderr," -m m  : kill client if gets more than this many MB behind, default %d\n", DEFMAXQSIZ);
	fprintf (stderr," -n    : ignore %s\n", lockout_fn);
	fprintf (stderr," -p p  : alternate IP port, default %d\n", INDIPORT);
	fprintf (stderr," -v    : show key events, no traffic\n");
	fprintf (stderr," -vv   : -v + key message content\n");
	fprintf (stderr," -vvv  : -vv + complete xml\n");
	fprintf (stderr," -x    : exit after last client disconnects -- FOR PROFILING ONLY\n");
	fprintf (stderr,"driver : executable or device@host[:port]\n");

	exit (2);
}

/* unlimit core dumps */
static void
allowCoreDumps()
{
	struct rlimit corelim;

	corelim.rlim_cur = RLIM_INFINITY;
	corelim.rlim_max = RLIM_INFINITY;
	if (setrlimit (RLIMIT_CORE, &corelim) < 0)
	    fprintf (stderr, "Can not allow cores: %s\n", strerror(errno));
}

/* turn off SIGPIPE on bad write so we can handle it inline */
static void
noSIGPIPE()
{
	struct sigaction sa;
	memset (&sa, 0, sizeof(sa));
	sa.sa_handler = SIG_IGN;
	sigemptyset(&sa.sa_mask);
	(void)sigaction(SIGPIPE, &sa, NULL);
}

/* start a thread that starts the given driver the first time.
 * N.B. only use this the first time, use restartDvr for any subsequent restarts.
 */
static void
initDvr (DvrInfo *dp, char *name)
{
	pthread_attr_t attr;
	pthread_t thr;

	/* save name */
	dp->name = name;

	/* init this thread's restart lock */
	pthread_rwlock_init (&dp->restart_lock, NULL);

	/* new thread will be detached so we need no join */
	if (pthread_attr_init (&attr))
	    Bye ("Driver %s attr init: %s\n", dp->name, strerror(errno));
	if (pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED))
	    Bye ("Driver %s setdetacthed: %s\n", dp->name, strerror(errno));

	/* start the thread that starts this driver */
	if (pthread_create (&thr, &attr, startDvrThread, (void*)dp))
	    Bye ("Driver %s startDvrThread thread: %s\n", dp->name, strerror(errno));
}

/* thread that just runs startDvr and exits.
 */
static void *
startDvrThread (void *vp)
{
	DvrInfo *dp = (DvrInfo *)vp;

	/* lock while setting up driver */
	pthread_rwlock_wrlock (&dp->restart_lock);
	startDvr (dp);
	pthread_rwlock_unlock (&dp->restart_lock);

	return(0);	/* thread exit */
}

/* start the given INDI driver process or connection.
 * wait a while if too fast, exit if trouble.
 * N.B. we assume restart_lock is already write-locked.
 */
static void
startDvr (DvrInfo *dp)
{
	time_t now = time(NULL);
	long age = now - dp->start;

	if (age < RESTARTDT) {
	    unsigned int sdt = RESTARTDT - age;
	    logMessage ("Driver %s: delaying restart by %d secs, min restart interval is %d secs\n",
	    	dp->name, sdt, RESTARTDT);
	    ssleep (sdt*1000);
	}
	dp->start = time(NULL);

	if (strchr (dp->name, '@'))
	    startRemoteDvr (dp);
	else
	    startLocalDvr (dp);
}

/* start the given local INDI driver process.
 * exit if trouble.
 * N.B. we assume restart_lock is already write-locked.
 */
static void
startLocalDvr (DvrInfo *dp)
{
	pthread_attr_t attr;
	pthread_t thr;
	Msg *mp;
	char buf[1024];
	int rp[2], wp[2], ep[2];
	int pid, l;
	FILE *fp;

	/* wait while lockout file exists */
	while (!ignore_lockout && (fp = fopen (lockout_fn, "r")) != NULL) {
	    fclose (fp);
	    logMessage ("Sleeping %d secs because %s exists\n", RDRTIME, lockout_fn);
	    ssleep (RDRTIME*1000);
	}

	/* build three pipes: r, w and error */
	if (pipe (rp) < 0)
	    Bye ("Driver %s read pipe: %s\n", dp->name, strerror(errno));
	if (pipe (wp) < 0)
	    Bye ("Driver %s write pipe: %s\n", dp->name, strerror(errno));
	if (pipe (ep) < 0)
	    Bye ("Driver %s stderr pipe: %s\n", dp->name, strerror(errno));

	/* fork&exec new process, connect pipes */
	pid = fork();
	if (pid < 0)
	    Bye ("driver %s fork error: %s\n", dp->name, strerror(errno));
	if (pid == 0) {
	    /* child: exec name */
	    int fd;

	    /* rig up pipes */
	    dup2 (wp[0], 0);	/* driver stdin reads from wp[0] */
	    dup2 (rp[1], 1);	/* driver stdout writes to rp[1] */
	    dup2 (ep[1], 2);	/* driver stderr writes to e[]1] */
	    for (fd = 3; fd < 100; fd++)
		(void) close (fd);

	    /* go -- should never return */
	    execlp (dp->name, dp->name, NULL);
	    logMessage ("Driver %s: execlp: %s\n", dp->name, strerror(errno));
	    _exit (EXITEXFAIL);	/* parent will notice EOF shortly */
	}

	/* don't need child's side of pipes */
	close (wp[0]);
	close (rp[1]);
	close (ep[1]);

	/* record pid, io channels, init lp, locks and snoop list */
	dp->pid = pid;
	dp->rfd = rp[0];
	dp->wfd = wp[1];
    dp->efd = ep[0];
	dp->err = 0;
	dp->lp = newLilXML();
	dp->mp = newMsg();
	dp->msgq = newFQ(1);
	pthread_mutex_init (&dp->q_lock, NULL);
	pthread_cond_init (&dp->go_cond, NULL);
	pthread_rwlock_init (&dp->sprops_rwlock, NULL);
	dp->sprops = (Snoopee**) malloc (1);	/* seed for realloc */
	if (!dp->sprops)
	    Bye ("No memory to seed sprops starting local driver %s\n", dp->dev);
	dp->nsprops = 0;

	if (verbose > 0)
	    logMessage ("Driver %s: pid=%d rfd=%d wfd=%d efd=%d\n",
			    dp->name, dp->pid, dp->rfd, dp->wfd, ep[0]);

	/* start detached threads */
	if (pthread_attr_init (&attr))
	    Bye ("Driver %s attr init: %s\n", dp->name, strerror(errno));
	if (pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED))
	    Bye ("Driver %s setdetacthed: %s\n", dp->name, strerror(errno));
	(void) pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	if (pthread_create (&thr, &attr, driverStdoutReaderThread, dp))
	    Bye ("Driver %s stdout thread: %s\n", dp->name, strerror(errno));
	if (pthread_create (&dp->stderr_thr, &attr, driverStderrReaderThread, dp))
	    Bye ("Driver %s stderr thread: %s\n", dp->name, strerror(errno));
	if (pthread_create (&thr, &attr, driverWriterThread, dp))
	    Bye ("Driver %s stdin thread: %s\n", dp->name, strerror(errno));
	if (pthread_attr_destroy (&attr))
	    Bye ("Driver %s attr destroy: %s\n", dp->name, strerror(errno));

	/* first message primes driver to report its properties -- dev already
	 * known if just restarting
	 */
	mp = newMsg();
	if (dp->dev[0])
	    l = snprintf (buf, sizeof(buf), "<getProperties device='%s' version='%g'/>\n", dp->dev, INDIV);
	else
	    l = snprintf (buf, sizeof(buf), "<getProperties version='%g'/>\n", INDIV);
	addMsg (mp, buf, l);
	(void) pushMsg (dp, NULL, mp);
	decMsg (mp);
}

/* start the given remote INDI driver connection.
 * repeat until socket opens ok, loop only blocks this thread.
 * N.B. we assume restart_lock is already write-locked.
 */
static void
startRemoteDvr (DvrInfo *dp)
{
	pthread_attr_t attr;
	pthread_t thr;
	socklen_t len = sizeof(dp->addr);
	Msg *mp;
	char dev[1024];
	char host[1024];
	char buf[1024];
	int port, sockfd;
	int l;

	/* extract host and port */
	port = INDIPORT;
	if (sscanf (dp->name, "%1023[^@]@%1023[^:]:%d", dev, host, &port) < 2)
	    Bye ("Bad remote device syntax: %s\n", dp->name);

	/* try connect forever until success */
	while (1) {
	    sockfd = openRemoteConnection (host, port);
	    if (sockfd < 0) {
	        logMessage ("Sleeping %d secs to retry %s\n", RDRTIME, dp->name);
		ssleep (RDRTIME*1000);
	    } else
		break;
	}

	/* record flag pid, io channels, init lp, locks and snoop list */
	dp->pid = REMOTEDVR;
	memset (&dp->addr, 0, sizeof(dp->addr));
	getpeername(sockfd, (struct sockaddr*)&dp->addr, &len);
	strcpy (dp->addrname, inet_ntoa (dp->addr.sin_addr));
	dp->rfd = sockfd;
	dp->wfd = sockfd;
	dp->err = 0;
	dp->lp = newLilXML();
	dp->mp = newMsg();
	dp->msgq = newFQ(1);
	pthread_mutex_init (&dp->q_lock, NULL);
	pthread_cond_init (&dp->go_cond, NULL);
	pthread_rwlock_init (&dp->sprops_rwlock, NULL);
	dp->sprops = (Snoopee**) malloc (1);	/* seed for realloc */
	if (!dp->sprops)
	    Bye ("No memory to seed sprops starting remore driver %s\n", dev);
	dp->nsprops = 0;

	/* N.B. storing name now is key to limiting outbound traffic to this
	 * dev.
	 */
	strncpyz (dp->dev, dev, MAXINDIDEVICE-1);

	logMessage ("Driver %s at %s now connected on socket=%d\n", dp->name, dp->addrname, sockfd);

	/* start detached threads */
	if (pthread_attr_init (&attr))
	    Bye ("Driver %s attr init: %s\n", dp->name, strerror(errno));
	if (pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED))
	    Bye ("Driver %s setdetacthed: %s\n", dp->name, strerror(errno));
	(void) pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	if (pthread_create (&thr, &attr, driverStdoutReaderThread, dp))
	    Bye ("Driver %s stdout thread: %s\n", dp->name, strerror(errno));
	if (pthread_create (&thr, &attr, driverWriterThread, dp))
	    Bye ("Driver %s stdin thread: %s\n", dp->name, strerror(errno));
	if (pthread_attr_destroy (&attr))
	    Bye ("Driver %s attr destroy: %s\n", dp->name, strerror(errno));

	/* Sending getProperties with device lets remote server limit its
	 * outbound (and our inbound) traffic on this socket to this device.
	 */
	mp = newMsg();
	l = snprintf (buf,sizeof(buf), "<getProperties device='%s' version='%g'/>\n", dp->dev, INDIV);
	addMsg (mp, buf, l);
	(void) pushMsg (dp, NULL, mp);
	decMsg(mp);

	/* This should work like a driver, ie, we always get all its BLOBs.
	 * Then here we honor enableBLOB from each of our clients.
	 */
	mp = newMsg();
	l = snprintf (buf, sizeof(buf), "<enableBLOB device='%s'>Also</enableBLOB>\n", dp->dev);
	addMsg (mp, buf, l);
	(void) pushMsg (dp, NULL, mp);
	decMsg(mp);
}

/* connect to a remote driver, probably an indiserver but could be a socket-based driver,
 * at the given host and port or die.
 * return socket fd if ok, exit if basic problem, return -1 if host not responding.
 */
static int
openRemoteConnection (char host[], int port)
{
	struct addrinfo hints, *aip;
	char port_str[16];
	int sockfd;
	int sockopt;
	socklen_t optlen = sizeof(sockopt);

	/* lookup host address.
	 * N.B. must call freeaddrinfo(aip) after successful call before returning
	 */
	memset (&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	snprintf (port_str, sizeof(port_str), "%d", port);
	int error = getaddrinfo (host, port_str, &hints, &aip);
	if (error) {
	    logMessage ("getaddrinfo(%s:%d): %s\n", host, port, gai_strerror(error));
	    return (-1);
	}

	/* create socket */
	sockfd = socket (aip->ai_family, aip->ai_socktype, aip->ai_protocol);
	if (sockfd < 0) {
	    freeaddrinfo (aip);
	    Bye ("socket(%s:%d): %s\n", host, port, strerror(errno));
	}

	/* use keep-alive to detect crashed "half-open" peers.
	 * see http://www.tldp.org/HOWTO/html_single/TCP-Keepalive-HOWTO
	 */
	sockopt = 1;
	if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &sockopt, optlen) < 0)
	    Bye ("setsockopt(SO_KEEPALIVE) on %s:%d: %s\n", strerror(errno));
	sockopt = 0;
	if (getsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &sockopt, &optlen) < 0)
	    Bye ("getsockopt(SO_KEEPALIVE) on %s:%d: %s\n", strerror(errno));
	if (!sockopt)
	    Bye ("SO_KEEPALIVE for %s:%d not confirmed: %s\n", strerror(errno));

#ifdef TCP_KEEPIDLE      // linux
	sockopt = 5;	// number of keepalive probes before reporting failure
	if (setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT, &sockopt, optlen) < 0)
	    Bye ("setsockopt(TCP_KEEPCNT) on %s:%d: %s\n", strerror(errno));
	sockopt = 10;	// seconds before first keepalive probe
	if (setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE, &sockopt, optlen) < 0)
	    Bye ("setsockopt(TCP_KEEPIDLE) on %s:%d: %s\n", strerror(errno));
	sockopt = 2;	// seconds between subsequent probes
	if (setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, &sockopt, optlen) < 0)
	    Bye ("setsockopt(TCP_KEEPINTVL) on %s:%d: %s\n", strerror(errno));
#endif

#ifdef TCP_KEEPALIVE      // macos
	sockopt = 10;	// keep alive seconds
	if (setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPALIVE, &sockopt, optlen) < 0)
	    Bye ("setsockopt(TCP_KEEPCNT) on %s:%d: %s\n", strerror(errno));
#endif

	/* connect */
	if (connect (sockfd, aip->ai_addr, aip->ai_addrlen) < 0) {
	    logMessage ("connect(%s,%d): %s\n", host,port,strerror(errno));
	    freeaddrinfo (aip);
	    close (sockfd);
	    return (-1);
	}

	/* ok */
	freeaddrinfo (aip);
	return (sockfd);
}

/* create our public indiserver endpoint lsocket on port.
 * return server socket else exit.
 */
static void
indiListen ()
{
	struct sockaddr_in serv_socket;
	int sfd;
	int reuse = 1;

	/* make socket endpoint */
	if ((sfd = socket (AF_INET, SOCK_STREAM, 0)) < 0)
	    Bye ("socket: %s\n", strerror(errno));

	/* bind to given port for any IP address */
	memset (&serv_socket, 0, sizeof(serv_socket));
	serv_socket.sin_family = AF_INET;
	serv_socket.sin_addr.s_addr = htonl (INADDR_ANY);
	serv_socket.sin_port = htons ((unsigned short)port);
	if (setsockopt(sfd,SOL_SOCKET,SO_REUSEADDR,&reuse,sizeof(reuse)) < 0)
	    Bye ("setsockopt: %s\n", strerror(errno));
	if (bind(sfd,(struct sockaddr*)&serv_socket,sizeof(serv_socket)) < 0)
	    Bye ("bind: %s\n", strerror(errno));

	/* willing to accept connections with a backlog of 5 pending */
	if (listen (sfd, 50) < 0)
	    Bye ("listen: %s\n", strerror(errno));

	/* ok */
	lsocket = sfd;
	if (verbose > 0)
	    logMessage ("listening to port %d on fd %d\n", port, sfd);
}

/* prepare for new client arriving on lsocket.
 * exit if trouble.
 */
static void
newClient()
{
	pthread_attr_t attr;
	pthread_t thr;
	ClInfo *cp = NULL;
	socklen_t len = sizeof(struct sockaddr_in);
	int s, i;

	/* assign new socket */
	s = newClSocket ();

	/* lock clinfo for changes */
	pthread_rwlock_wrlock (&cl_rwlock);

	if (verbose > 2) {
	    int nactive;
	    for (nactive = i = 0; i < nclinfo; i++)
		if (clinfo[i]->active)
		    nactive++;
	    logMessage ("newClient() starting with %d clinfo %d active\n", nclinfo, nactive);
	}

	/* try to reuse a clinfo slot, else add one */
	for (i = 0; i < nclinfo; i++)
	    if (!(cp = clinfo[i])->active)
		break;
	if (i == nclinfo) {
	    /* grow clinfo */
	    clinfo = (ClInfo **) realloc (clinfo, (nclinfo+1)*sizeof(ClInfo*));
	    if (!clinfo)
		Bye ("no memory for new client table\n");
	    clinfo[nclinfo++] = cp = (ClInfo *) malloc (sizeof(ClInfo));
	    if (!cp)
		Bye ("no memory for new client\n");
	}

	/* rig up new clinfo entry */
	memset (cp, 0, sizeof(*cp));
	cp->active = 1;
	cp->s = s;
	cp->lp = newLilXML();
	cp->mp = newMsg();
	cp->msgq = newFQ(1);
	pthread_mutex_init (&cp->q_lock, NULL);
	pthread_cond_init (&cp->go_cond, NULL);
	pthread_rwlock_init (&cp->props_rwlock, NULL);
	cp->props = (Property *) malloc (1);
	if (!cp->props)
	    Bye ("No props memory for new client\n");
	cp->blobs = (Property *) malloc (1);
	if (!cp->blobs)
	    Bye ("No blobs memory for new client\n");
	getpeername(s, (struct sockaddr*)&cp->addr, &len);
	strcpy (cp->addrname, inet_ntoa (cp->addr.sin_addr));

	/* done changing clinfo */
	pthread_rwlock_unlock (&cl_rwlock);

	if (verbose > 0) {
	    logMessage ("Client %d: new arrival from %s:%d - hello!\n",
			cp->s, cp->addrname, ntohs(cp->addr.sin_port));
	}

	/* start detached threads */
	if (pthread_attr_init (&attr))
	    Bye ("Client attr init: %s\n", strerror(errno));
	if (pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED))
	    Bye ("Client setdetacthed: %s\n", strerror(errno));
	(void) pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	if (pthread_create (&thr, &attr, clientReaderThread, cp))
	    Bye ("Client read thread: %s\n", strerror(errno));
	if (pthread_create (&thr, &attr, clientWriterThread, cp))
	    Bye ("Client write thread: %s\n", strerror(errno));
	if (pthread_attr_destroy (&attr))
	    Bye ("Client attr destroy: %s\n", strerror(errno));
}

/* block to accept a new client arriving on lsocket.
 * return private socket or exit.
 */
static int
newClSocket ()
{
	struct sockaddr_in cli_socket;
	socklen_t cli_len;
	int cli_fd;

	/* get a private connection to new client */
	cli_len = sizeof(cli_socket);
	cli_fd = accept (lsocket, (struct sockaddr *)&cli_socket, &cli_len);
	if(cli_fd < 0)
	    Bye ("accept: %s\n", strerror(errno));

	/* ok */
	return (cli_fd);
}

/* thread to read from the given client, send to each appropriate driver when see
 * xml closure. also send all newXXX() to all other interested clients.
 * if trouble signal clientWriterThread and return.
 */
static void *
clientReaderThread (void *vp)
{
	ClInfo *cp = (ClInfo *)vp;
	int i, nr;

	/* read until client disconnects */
	while (1) {
	    /* insure more message space */
	    minMsg (cp->mp, MAXRBUF);

	    /* read more from client directly into cp->mp */
	    nr = read (cp->s, cp->mp->cp + cp->mp->used, cp->mp->total - cp->mp->used);
	    if (nr <= 0) {
		if (nr < 0)
		    logMessage ("from Client %d: read error: %s\n", cp->s, strerror(errno));
		else if (verbose > 0)
		    logMessage ("from Client %d: read EOF\n", cp->s);
		onClientError (cp);
		return (NULL);	/* thread exit */
	    }
	    cp->mp->used += nr;

	    /* process XML, sending when find closure */
	    for (i = 0; i < nr; i++) {
		char err[1024];
		XMLEle *root = readXMLEle (cp->lp, cp->mp->cp[cp->mp->next++], err);
		if (root) {
		    /* found new complete message */

		    char *roottag = tagXMLEle(root);
		    char *dev = findXMLAttValu (root, "device");
		    char *name = findXMLAttValu (root, "name");
		    int isblob = !strcmp (roottag, "setBLOBVector");
		    Msg *newmp;

		    /* keep the good part and start a new msg with remaining */
		    newmp = splitMsg (cp->mp, cp->mp->next);

		    if (verbose > 3) {
			logMessage ("from Client %d: read:\n", cp->s);
			traceMsg (root);
		    } else if (verbose > 2) {
			logMessage ("from Client %d: read <%s device='%s' name='%s'>\n",
					cp->s, roottag, dev, name);
		    } else if (verbose > 1)
			logMsg ("from", NULL, cp, cp->mp);

		    /* enableBLOB control is just handled locally. */
		    if (!strcmp (roottag, "enableBLOB")) {
			BLOBHandling bh;
			crackBLOB (pcdataXMLEle(root), &bh);
			if (bh == B_ALSO || bh == B_ONLY)
			    addClDevice (cp, 1, dev, name);
			else
			    rmClDevice (cp, 1, dev, name);
			goto done;
		    }

		    /* snag interested properties */
		    addClDevice (cp, 0, dev, name);

		    /* send message to driver(s) responsible for dev */
		    q2Drivers (dev, cp->mp, roottag);

		    /* echo new* commands back to other clients */
		    if (!strncmp (roottag, "new", 3))
			q2Clients (cp, isblob, dev, name, cp->mp);

		  done:

		    /* we're done with this msg here */
		    decMsg (cp->mp);

		    /* continue with newmp */
		    cp->mp = newmp;

		    /* done with root */
		    delXMLEle (root);

		} else if (err[0]) {
		    logMessage ("from Client %d: XML error: %s\n", cp->s, err);
		    onClientError (cp);
		    return (NULL);	/* thread exit */
		}
	    }
	}

	/* for lint */
	return (NULL);
}

/* thread to send Msgs to the given client.
 * wait for CV, pop message from queue, send and free if we are the last user.
 * shut down this client and return if trouble.
 */
static void *
clientWriterThread (void *vp)
{
	ClInfo *cp = (ClInfo *)vp;
	int nsent, nsend, nw;
	Msg *mp;


	while (1) {

	    /* lock our q */
	    pthread_mutex_lock (&cp->q_lock);

	    /* wait while queue is empty or no errors detected */
	    while (nFQ(cp->msgq) == 0 && !cp->err)
		pthread_cond_wait (&cp->go_cond, &cp->q_lock);

	    if (cp->err) {

		pthread_mutex_unlock (&cp->q_lock);
		shutdownClient (cp);
		return (NULL);	/* thread exit */

	    } else {

		/* get next message */
		mp = (Msg *) popFQ (cp->msgq);
		if (!mp)
		    Bye ("Bug! Client %d message queue is empty!\n", cp->s);
		if (verbose > 1)
		    logMsg ("send to", NULL, cp, mp);

		/* ok to let others q us more msgs while we send this one */
		pthread_mutex_unlock (&cp->q_lock);

		/* send */
		for (nsent = 0; nsent < mp->used; nsent += nw) {
		    nsend = mp->used - nsent;
		    if (nsend > MAXWSIZ)
			nsend = MAXWSIZ;
		    nw = write (cp->s, mp->cp+nsent, nsend);
		    if (nw <= 0) {
			if (nw == 0)
			    logMessage ("to Client %d: write returned 0 with %d on q\n", cp->s, nFQ(cp->msgq));
			else if (verbose > 1 || errno != EPIPE) {
			    /* EPIPE errors are not reported because they are
			     * too numerous to be interesting as we wait for
			     * clientReader to detect problem and set dp->err
			     */
			    logMessage ("to Client %d: write with %d on q: %s\n", cp->s, nFQ(cp->msgq),
			    							strerror(errno));
			}

			/* give up, let reader thread discover pipe error and set cp->err */
			break;
		    }
		}

		/* one more nl to help DOM parsers */
		(void) write (cp->s, "\n", 1);

		/* finished with this message, even if error sending */
		decMsg (mp);
	    }
	}

	/* for lint */
	return (NULL);
}

/* thread to read from the given local driver's stdout or remote driver's socket.
 * send messages to each interested client when see xml closure.
 * if trouble signal driverWriterThread and return/exit.
 */
static void *
driverStdoutReaderThread (void *vp)
{
	DvrInfo *dp = (DvrInfo *)vp;
	int i, nr;

	while (1) {
	    /* insure more message space */
	    minMsg (dp->mp, MAXRBUF);

	    /* read more from driver */
	    nr = read (dp->rfd, dp->mp->cp + dp->mp->used, dp->mp->total - dp->mp->used);
	    if (nr <= 0) {
		if (nr < 0)
		    logMessage ("from Driver %s: stdin %s\n", dp->name, strerror(errno));
		else
		    logMessage ("from Driver %s: stdin EOF\n", dp->name);
		onDriverError (dp);
		return (NULL);	/* thread exit */
	    }
	    dp->mp->used += nr;

	    /* process XML, sending when find closure */
	    for (i = 0; i < nr; i++) {
		char err[1024];
		XMLEle *root = readXMLEle (dp->lp, dp->mp->cp[dp->mp->next++], err);
		if (root) {
		    /* found new complete message */

		    char *roottag = tagXMLEle(root);
		    char *dev = findXMLAttValu (root, "device");
		    char *name = findXMLAttValu (root, "name");
		    int isblob = !strcmp (roottag, "setBLOBVector");
		    Msg *newmp;

		    /* keep the good part and start a new msg with remaining */
		    newmp = splitMsg (dp->mp, dp->mp->next);

		    if (verbose > 3) {
			logMessage ("from Driver %s: read:\n", dp->name);
			traceMsg (root);
		    } else if (verbose > 2) {
			logMessage ("from Driver %s: read <%s device='%s' name='%s'>\n",
					dp->name, roottag, dev, name);
		    } else if (verbose > 1)
			logMsg ("from", dp, NULL, dp->mp);

		    /* that's all if driver is just registering a snoop */
		    if (!strcmp (roottag, "getProperties")) {
			addSnoopDevice (dp, dev, name);
                        q2Drivers (dev, dp->mp, roottag);        // force initial report
			goto done;
		    }

		    /* that's all if driver is just registering a BLOB mode */
		    if (!strcmp (roottag, "enableBLOB")) {
			Snoopee *sp = findSnoopDevice (dp, dev, name);
			if (sp)
			    crackBLOB (pcdataXMLEle (root), &sp->blob);
			goto done;
		    }

		    /* snag device name if not known yet */
		    if (!dp->dev[0] && dev[0]) {
			strncpyz (dp->dev, dev, MAXINDIDEVICE-1);
			if (verbose > 1)
			    logMessage ("Driver %s snooping for %s\n", dp->name, dp->dev);
		    }

		    /* log messages if any */
		    logDvrMsg (root, dev);

		    /* send to interested clients */
		    q2Clients (NULL, isblob, dev, name, dp->mp);

		    /* send to snooping drivers */
		    q2SnoopingDrivers (isblob, dev, name, dp->mp);

		done:

		    /* we're done with this msg here */
		    decMsg (dp->mp);

		    /* continue with newmp */
		    dp->mp = newmp;

		    /* done with root */
		    delXMLEle (root);

		} else if (err[0]) {
		    logMessage ("Driver %s: XML error: %s\n", dp->name, err);
		    onDriverError (dp);
		    return (NULL);	/* thread exit */
		}
	    }
	}

	/* for lint */
	return (NULL);
}

/* thread to read from the given local driver's stderr.
 * read lines and add prefix then send to our log file.
 * just return if trouble, let driverStdoutReaderThread inform writer.
 * we get rudely cancelled if this driver gets into trouble.
 */
static void *
driverStderrReaderThread (void *vp)
{
	DvrInfo *dp = (DvrInfo *)vp;
	char buf[MAXRBUF];
	int oldstate;

	/* make sure we are cancellable */
	pthread_setcancelstate (PTHREAD_CANCEL_ENABLE, &oldstate);

	/* log everthing until error */
	while (1) {
	    /* read next whole line */
        ssize_t rv = read(dp->efd, buf, sizeof(buf)-1);
        if(rv == 0)
        {
            logMessage ("from Driver %s: stderr EOF\n", dp->name);
            return NULL;
        }
        else if(rv < 0)
        {
            logMessage ("from Driver %s: stderr %s\n", dp->name, strerror(errno));
            return NULL;
        }
        buf[rv] = '\0';

	    /* prefix each whole line to our stderr, save extra for next time */
	    logMessage ("Driver %s: %s", dp->name, buf);	/* includes nl */
	}
    

	/* for lint */
	return (NULL);
}

/* thread to send Msgs to the given local driver.
 * wait for CV, pop message from queue, send and free if we are the last user.
 * restart this driver if trouble.
 */
static void *
driverWriterThread (void *vp)
{
	DvrInfo *dp = (DvrInfo *)vp;
	int nsent, nsend, nw;
	Msg *mp;

	while (1) {

	    /* lock our q */
	    pthread_mutex_lock (&dp->q_lock);

	    /* wait while queue is empty or no errors detected */
	    while (nFQ(dp->msgq) == 0 && !dp->err)
		pthread_cond_wait (&dp->go_cond, &dp->q_lock);

	    if (dp->err) {

		pthread_mutex_unlock (&dp->q_lock);
		restartDvr(dp);
		return (NULL);	/* thread exit */

	    } else {

		/* get next message */
		mp = (Msg *) popFQ (dp->msgq);
		if (!mp)
		    Bye ("Bug! Driver %s message queue is empty!\n", dp->name);
		if (verbose > 1)
		    logMsg ("send to", dp, NULL, mp);

		/* ok to let others q us more msgs while we send this one */
		pthread_mutex_unlock (&dp->q_lock);

		/* send */
		for (nsent = 0; nsent < mp->used; nsent += nw) {
		    nsend = mp->used - nsent;
		    if (nsend > MAXWSIZ)
			nsend = MAXWSIZ;
		    nw = write (dp->wfd, mp->cp+nsent, nsend);
		    if (nw <= 0) {
			if (nw == 0)
			    logMessage ("to Driver %s: write returned 0 with %d on q\n", dp->name, nFQ(dp->msgq));
			else if (verbose > 1 || errno != EPIPE) {
			    /* EPIPE errors are not reported because they are
			     * too numerous to be interesting as we wait for
			     * driverStdinReader to detect problem and set dp->err
			     */
			    logMessage ("to Driver %s: write with %d on q: %s\n", dp->name, nFQ(dp->msgq), strerror(errno));
			}

			/* give up, let reader thread discover pipe error and set dp->err */
			break;
		    }
		}

		/* one more nl to help DOM parsers */
		(void) write (dp->wfd, "\n", 1);

		/* finished with this message, even if error sending */
		decMsg (mp);
	    }
	}

	/* for lint */
	return (NULL);
}

/* called by driverStdoutReaderThread to inform driverWriterThread it has
 * detected an error
 */
static void
onDriverError (DvrInfo *dp)
{
	logMessage ("Driver %s: reader thread indicates it's time to restart\n", dp->name);
	pthread_mutex_lock (&dp->q_lock);
	logMessage ("Driver %s: locked mutex to set error condition\n", dp->name);
	dp->err = 1;
	pthread_cond_signal (&dp->go_cond);
	pthread_mutex_unlock (&dp->q_lock);
	logMessage ("Driver %s: reader thread signaled go_cond and unlocked q_lock\n", dp->name);
}

/* called by clientReaderThread to inform clientWriterThread it has
 * detected an error
 */
static void
onClientError (ClInfo *cp)
{
	pthread_mutex_lock (&cp->q_lock);
	cp->err = 1;
	pthread_cond_signal (&cp->go_cond);
	pthread_mutex_unlock (&cp->q_lock);
}

/* close down the given client.
 */
static void
shutdownClient (ClInfo *cp)
{
	/* lock clinfo while updating */
	pthread_rwlock_wrlock (&cl_rwlock);

	/* close socket connection */
	shutdown (cp->s, SHUT_RDWR);
	close (cp->s);

	/* free memory and locks */
	delLilXML (cp->lp);
	free (cp->props);
	free (cp->blobs);
	decMsg (cp->mp);
	pthread_mutex_destroy (&cp->q_lock);
	pthread_cond_destroy (&cp->go_cond);
	pthread_rwlock_destroy (&cp->props_rwlock);
	if (verbose > 1)
	    logMessage ("Client %d: draining with %d on queue\n", cp->s, nFQ(cp->msgq));
	drainMsgs (cp->msgq);
	delFQ (cp->msgq);

	if (verbose > 0)
	    logMessage ("Client %d: shut down complete - good-bye!\n", cp->s);

	/* ok now to recycle -- also sets active = 0 */
	memset (cp, 0, sizeof(*cp));

	/* this is just used when profiling with gprof and friends */
	if (profile_exit) {
	    int i, n;
	    for (n = i = 0; i < nclinfo; i++)
		if (clinfo[i]->active)
		    n++;
	    if (n == 0) {
		logMessage ("Closing after last client disconnected to support profiling\n");
		exit(0);
	    }
	}

	/* done */
	pthread_rwlock_unlock (&cl_rwlock);
}

/* close down the given driver and restart.
 * N.B. lock restart_lock so no other threads try to use it until ready to go again.
 */
static void
restartDvr (DvrInfo *dp)
{
	int i;

	/* write-lock while we edit */
	pthread_rwlock_wrlock (&dp->restart_lock);

	/* make sure it's dead, reclaim resources */
	if (dp->pid == REMOTEDVR) {
	    /* socket connection */
	    shutdown (dp->wfd, SHUT_RDWR);
	    close (dp->wfd);	/* same as rfd */
	} else {
	    /* local pipe connection broke */
	    int status, wpid = waitpid (dp->pid, &status, 0);
	    if (wpid == dp->pid) {
		if (WIFEXITED(status)) {
		    int es = WEXITSTATUS(status);
		    logMessage ("Driver %s: Exit status %d\n", dp->name, es);
		    if (es == EXITEXFAIL) {
			logMessage ("Exiting because of hopeless driver: %s\n", dp->name);
			exit(1);
		    }
		} else if (WIFSIGNALED(status))
#ifdef WCOREDUMP
		    logMessage ("Driver %s: Exit signal %d%s\n", dp->name,
			WTERMSIG(status),
			WCOREDUMP(status) ? " (core dumped)" : "");
#else
		    logMessage ("Driver %s: Exit signal %d\n", dp->name,
			WTERMSIG(status));
#endif	/* WCOREDUMP */
		else
		    logMessage ("Driver %s: Unknown exit condition\n", dp->name);
	    } else {
			logMessage ("Driver %s: Killed: %d\n", dp->name,
			kill (dp->pid, SIGKILL));
			(void) waitpid (dp->pid, &status, WNOHANG);
	    }
		//int thispid = (int)gettid();
		logMessage ("Driver %s: closing dp->efd", dp->name);
	    close (dp->efd);
		logMessage ("Driver %s: closing dp->wfd", dp->name);
	    close (dp->wfd);
		logMessage ("Driver %s: closing dp->rfd", dp->name);
	    close (dp->rfd);
	}

	/* free memory and locks */
	logMessage ("Driver %s: freeing memory and locks\n", dp->name);
	for (i = 0; i < dp->nsprops; i++)
	    free (dp->sprops[i]);
	free (dp->sprops);
	delLilXML (dp->lp);
	decMsg (dp->mp);
	pthread_mutex_destroy (&dp->q_lock);
	pthread_cond_destroy (&dp->go_cond);
	pthread_rwlock_destroy (&dp->sprops_rwlock);
	// if (verbose > 1)
	    logMessage ("Driver %s: draining with %d on queue\n", dp->name, nFQ(dp->msgq));
	drainMsgs (dp->msgq);
	delFQ (dp->msgq);

	/* start this driver again */
	logMessage ("Driver %s: restart #%d\n", dp->name, ++dp->restarts);
	startDvr (dp);

	/* done */
	pthread_rwlock_unlock (&dp->restart_lock);
}

/* put Msg mp on queue of each driver responsible for dev, or all drivers
 *   if dev not specified.
 * N.B. add device to any generic getProperties going to remote drivers, else
 *   they get sent back out everywhere and go around forever.
 */
static void
q2Drivers (char *dev, Msg *mp, char *roottag)
{
	int isggp = !strcmp (roottag, "getProperties") && !dev[0];
	DvrInfo *dp;

	/* queue message to each driver unless it is restarting or we
	 * know it does not support this dev
	 */
	for (dp = dvrinfo; dp < &dvrinfo[ndvrinfo]; dp++) {

	    if (pthread_rwlock_tryrdlock (&dp->restart_lock) == 0) {

		if (!dev[0] || !dp->dev[0] || !strcmp (dev, dp->dev)) {

		    Msg *remote_mp = NULL;
		    Msg *sendmp;
		    int ql;

		    /* insure getProperties to remote drivers includes device to avoid
		     * chained loops
		     */
		    if (isggp && dp->pid == REMOTEDVR) {
			char gp[100];
			int gpl;

			if (verbose)
			    logMessage ("Driver %s: Loop caught, adding %s to generic getProperties\n",
					    dp->name, dp->dev);
			remote_mp = newMsg();
			gpl = snprintf (gp, sizeof(gp), "<getProperties version='%g' device='%s' />\n", INDIV, dp->dev);
			addMsg (remote_mp, gp, gpl);
			sendmp = remote_mp;
		    } else
			sendmp = mp;

		    /* ok: queue message to this driver -- beware it getting too far behind */
                    if (verbose > 2)
                        logMsg ("queue to", dp, NULL, sendmp);
		    ql = pushMsg (dp, NULL, sendmp);
		    if (ql > maxqsiz) {
			logMessage ("Driver %s: %d bytes behind in %d messages, restarting\n",
							dp->name, ql, nFQ(dp->msgq));

			/* close reader socket to force driverStdoutReader to set err */
			close (dp->rfd);

			/* just blow away stderr reader, if we have one */
			if (dp->pid != REMOTEDVR)
			    pthread_cancel (dp->stderr_thr);
		    }

		    /* finished with remote_mp here if we used it */
		    if (remote_mp)
			decMsg (remote_mp);
		}

		/* done with this dvr */
		pthread_rwlock_unlock (&dp->restart_lock);
	    }
	}
}

/* put Msg mp on queue of each driver snooping dev/name.
 * if is BLOB always honor current mode.
 */
static void
q2SnoopingDrivers (int isblob, char *dev, char *name, Msg *mp)
{
	DvrInfo *dp;
	int ql;

	/* queue message to each driver if it is not restarting and
	 * it is snooping for this dev/name
	 */
	for (dp = dvrinfo; dp < &dvrinfo[ndvrinfo]; dp++) {

	    if (pthread_rwlock_tryrdlock (&dp->restart_lock) == 0) {

		Snoopee *sp = findSnoopDevice (dp, dev, name);

		/* nothing for dp if not snooping for dev/name or wrong BLOB mode */
		if (sp && !((isblob && sp->blob==B_NEVER) || (!isblob && sp->blob==B_ONLY))) {

		    /* ok: queue message to this driver -- beware it getting too far behind */
		    ql = pushMsg (dp, NULL, mp);
		    if (ql > maxqsiz) {
			logMessage ("Driver %s: %d bytes behind in %d messages, restarting\n",
						    dp->name, ql, nFQ(dp->msgq));

			/* close reader socket to force driverStdoutReader to set err */
			close (dp->rfd);

			/* just blow away stderr reader, if we have one */
			if (dp->pid != REMOTEDVR)
			    pthread_cancel (dp->stderr_thr);
		    }
		}

		/* done with this dvr */
		pthread_rwlock_unlock (&dp->restart_lock);
	    }
	}
}

/* add dev/name to dp's snooping list.
 * init with blob mode set to B_NEVER.
 */
static void
addSnoopDevice (DvrInfo *dp, char *dev, char *name)
{
	Snoopee *sp;

	/* no dups */
	sp = findSnoopDevice (dp, dev, name);
	if (sp)
	    return;

	/* write access */
	pthread_rwlock_wrlock (&dp->sprops_rwlock);

	/* add dev to sprops list */
	dp->sprops = (Snoopee**) realloc (dp->sprops, (dp->nsprops+1)*sizeof(Snoopee*));
	if (!dp->sprops)
	    Bye ("No memory to add %d snoop device to %s.%s\n", dp->nsprops+1, dev, name);
	sp = dp->sprops[dp->nsprops++] = (Snoopee *) calloc (1, sizeof(Snoopee));

	strncpyz (sp->prop.dev, dev, MAXINDIDEVICE-1);
	strncpyz (sp->prop.name, name, MAXINDINAME-1);
	sp->blob = B_NEVER;

	/* unlock */
	pthread_rwlock_unlock (&dp->sprops_rwlock);

	if (verbose)
	    logMessage ("Driver %s: snooping on %s.%s\n", dp->name, dev, name);
}

/* return Snoopee if dp is snooping dev/name, else NULL.
 */
static Snoopee *
findSnoopDevice (DvrInfo *dp, char *dev, char *name)
{
	Snoopee *foundsp = NULL;
	int i;

	/* read access */
	pthread_rwlock_rdlock (&dp->sprops_rwlock);

	for (i = 0; i < dp->nsprops; i++) {
	    Snoopee *sp = dp->sprops[i];
	    Property *pp = &sp->prop;
	    if (!strcmp (pp->dev, dev) &&
		    (!pp->name[0] || !strcmp(pp->name, name))) {
		foundsp = sp;
		break;
	    }
	}

	/* unlock */
	pthread_rwlock_unlock (&dp->sprops_rwlock);

	return (foundsp);
}

/* put Msg mp on queue of each client interested in dev/name, except notme.
 * if BLOB always honor current mode.
 */
static void
q2Clients (ClInfo *notme, int isblob, char *dev, char *name, Msg *mp)
{
	ClInfo *cp;
	int i, ql;

	/* read access */
	pthread_rwlock_rdlock (&cl_rwlock);

	/* queue message to each interested client */
	for (i = 0; i < nclinfo; i++) {
	    /* in use? notme? want this dev/name? blob? */
	    cp = clinfo[i];
	    if (!cp->active || cp == notme)
		continue;
	    if (findClDevice (cp, isblob, dev, name) < 0)
		continue;

	    /* ok: queue message to this client -- beware it getting too far behind */
            if (verbose > 2)
                logMsg ("queue to", NULL, cp, mp);
	    ql = pushMsg (NULL, cp, mp);
	    if (ql > maxqsiz) {
		logMessage ("Client %d: %d bytes behind in %d messages, shutting down\n",
					cp->s, ql, nFQ(cp->msgq));
		/* close socket to force clientReader to set err */
		shutdown (cp->s, SHUT_RDWR);
		close (cp->s);
	    }
	}

	/* unlock */
	pthread_rwlock_unlock (&cl_rwlock);
}


/* increment mp count then push it onto dp or cp's queue for writing.
 * while we have the q locked find the total size of its messages.
 */
static int
pushMsg (DvrInfo *dp, ClInfo *cp, Msg *mp)
{
	FQ *qp;
	pthread_mutex_t *lp;
	pthread_cond_t *vp;
	int n;

	/* get appropriate q and locks */
	if (dp) {
	    qp = dp->msgq;
	    lp = &dp->q_lock;
	    vp = &dp->go_cond;
	} else if (cp) {
	    qp = cp->msgq;
	    lp = &cp->q_lock;
	    vp = &cp->go_cond;
	} else
	    return (0);

	/* increment usage count */
	incMsg (mp);

	/* push onto this queue, handy time to get size too */
	pthread_mutex_lock (lp);
	pushFQ (qp, mp);
	n = msgQSize (qp);
	pthread_cond_signal (vp);
	pthread_mutex_unlock (lp);

	return (n);
}

/* log message mp associated with either dp or cp (not both) with a label.
 * label is typically "from" or "to".
 */
static void
logMsg (const char *label, DvrInfo *dp, ClInfo *cp, Msg *mp)
{
	XMLEle *root;
	char ynot[1024];
	char *roottag, *dev, *name, *pc;

	/* parse and pull apart a little bit */
	root = parseXML (mp->cp, ynot);
	if (!root)
	    return;
	roottag = tagXMLEle(root);
	dev = findXMLAttValu (root, "device");
	name = findXMLAttValu (root, "name");
	pc = pcdataXMLEle (root);

	/* print enough to be recognized */
	if (dp)
	    logMessage ("%s Driver %s: q depth %d, msg count %d: \"<%.4s %s.%s>%.10s\"\n",
			label, dp->name, nFQ(dp->msgq), mp->count,
			    roottag, dev[0] ? dev : "*", name[0] ? name : "*", pc);
	else if (cp)
	    logMessage ("%s Client %d: q depth %d, msg count %d: \"<%.4s %s.%s>%.10s\"\n",
			label, cp->s, nFQ(cp->msgq), mp->count,
			    roottag, dev[0] ? dev : "*", name[0] ? name : "*", pc);

	delXMLEle (root);
}

/* return total size of all Msqs on the given q */
static int
msgQSize (FQ *q)
{
	int i, l = 0;

	for (i = 0; i < nFQ(q); i++) {
	    Msg *mp = (Msg *) peekiFQ(q,i);
	    l += mp->used;
	}

	return (l);
}

/* return pointer to one new empty Msg,
 * counting us as the first user.
 */
static Msg *
newMsg (void)
{
	Msg *newmp = (Msg *) malloc(sizeof(Msg));
	if (!newmp)
	    Bye ("No memory for new Msg\n");
	newmp->count = 1;
	newmp->used = 0;
	newmp->next = 0;
	newmp->cp = newmp->buf;
	newmp->total = sizeof(newmp->buf);
	pthread_mutex_init (&newmp->count_lock, NULL);
	return (newmp);
}

/* add buf[bufl] to mp, growing if necessary
 */
static void
addMsg (Msg *mp, char buf[], int bufl)
{
	minMsg (mp, bufl);
	memcpy (mp->cp + mp->used, buf, bufl);
	mp->used += bufl;
}

/* increment mp count */
static void
incMsg (Msg *mp)
{
	pthread_mutex_lock (&mp->count_lock);
	mp->count += 1;
	pthread_mutex_unlock (&mp->count_lock);
}

/* decrement count, free if reaches 0.
 * N.B. on return mp is not valid if its count came in as 1 or less.
 */
static void
decMsg (Msg *mp)
{
	pthread_mutex_lock (&mp->count_lock);
	if (--mp->count <= 0) {
	    if (mp->cp != mp->buf)
		free (mp->cp);
	    pthread_mutex_destroy (&mp->count_lock);
	    free (mp);
	} else
	    pthread_mutex_unlock (&mp->count_lock);
}


/* insure mp has at least min unused.
 */
static void
minMsg (Msg *mp, int min)
{
	int unused = mp->total - mp->used;
	if (unused < min) {
	    int newtot = mp->used + 2*min;
	    if (mp->cp == mp->buf) {
		/* first time, use malloc and copy from buf */
		mp->cp = (char *) malloc (newtot);
		if (!mp->cp)
		    Bye ("No memory for new minMsg cp of %d\n", newtot);
		memcpy (mp->cp, mp->buf, mp->used);
	    } else {
		/* already from heap, just realloc */
		mp->cp = (char *) realloc (mp->cp, newtot);
		if (!mp->cp)
		    Bye ("No memory to grow cp in minMsg to %d\n", newtot);
	    }
	    mp->total = newtot;
	}
}

/* retain only nkeep in mp and return new Msg with remainder.
 */
static Msg *
splitMsg (Msg *mp, int nkeep)
{
	Msg *newmp = newMsg();
	addMsg (newmp, mp->cp + nkeep, mp->used - nkeep);
	mp->used = nkeep;
	return (newmp);
}

/* free all Msgs in the given q
 */
static void
drainMsgs (FQ *qp)
{
	Msg *mp;

	while ((mp = (Msg*) popFQ(qp)) != NULL)
	    decMsg (mp);	/* decrements count and frees at 0 */
}

/* return index of props[] or blobs[] if cp may be interested in dev/name
 * else -1
 */
static int
findClDevice (ClInfo *cp, int isblob, char *dev, char *name)
{
	Property *pa;
	int nprops, i;

	/* protect while scanning */
	pthread_rwlock_rdlock (&cp->props_rwlock);

	/* get appropriate Property list and length */
	if (isblob) {
	    pa = cp->blobs;
	    nprops = cp->nblobs;
	} else {
	    pa = cp->props;
	    nprops = cp->nprops;
	}

	for (i = 0; i < nprops; i++) {
	    Property *pp = &pa[i];
	    if ((!dev[0] || !pp->dev[0] || !strcmp (dev, pp->dev))
		    && (!name[0] || !pp->name[0] || !strcmp (name, pp->name)))
		break;
	}
	if (i == nprops)
	    i = -1;

	/* unlock and return */
	pthread_rwlock_unlock (&cp->props_rwlock);
	return (i);
}

/* add the given device and name to the list of props[] (or blobs[], if isblob)
 * client cp is interested in. Avoid dups.
 */
static void
addClDevice (ClInfo *cp, int isblob, char *dev, char *name)
{
	Property *pp;

	/* no dups */
	if (findClDevice (cp, isblob, dev, name) >= 0)
	    return;

	/* protect while modifying */
	pthread_rwlock_wrlock (&cp->props_rwlock);

	/* add */
	if (isblob) {
	    cp->blobs = (Property *) realloc (cp->blobs,
					    (cp->nblobs+1)*sizeof(Property));
	    if (!cp->blobs)
		Bye ("No memory to grow %d blobs for %s.%s\n", cp->nblobs+1, dev, name);
	    pp = &cp->blobs[cp->nblobs++];
	    if (verbose > 1)
		logMessage ("Client %d listening for BLOB %s.%s\n", cp->s, dev, name);
	} else {
	    cp->props = (Property *) realloc (cp->props,
					    (cp->nprops+1)*sizeof(Property));
	    if (!cp->props)
		Bye ("No memory to grow %d props for %s.%s\n", cp->nprops+1,dev, name);
	    pp = &cp->props[cp->nprops++];
	    if (verbose > 1)
		logMessage ("Client %d listening for %s.%s\n", cp->s,
					dev[0] ? dev : "*", name[0] ? name : "*");
	}

	strncpyz (pp->dev, dev, MAXINDIDEVICE-1);
	strncpyz (pp->name, name, MAXINDINAME-1);

	/* unlock and finished */
	pthread_rwlock_unlock (&cp->props_rwlock);
}

/* remove all properties from the props[] (or blobs[], if isblob) for client
 * cp matching the given dev and name.
 * harmless if absent.
 */
static void
rmClDevice (ClInfo *cp, int isblob, char *dev, char *name)
{
	int i;

	while ((i = findClDevice (cp, isblob, dev, name)) >= 0) {

	    /* protect while modifying */
	    pthread_rwlock_wrlock (&cp->props_rwlock);

	    if (isblob)
		memmove (&cp->blobs[i], &cp->blobs[i+1],
			    (--cp->nblobs - i)*sizeof(Property));
	    else
		memmove (&cp->props[i], &cp->props[i+1],
			    (--cp->nprops - i)*sizeof(Property));

	    /* don't bother to realloc smaller, also avoids dealing with empty array */

	    /* unlock */
	    pthread_rwlock_unlock (&cp->props_rwlock);
	}
}

/* convert the string value of enableBLOB to our B_ state value.
 * default to NEVER if unrecognized.
 */
static void
crackBLOB (char *enableBLOB, BLOBHandling *bp)
{
	if (!strcmp (enableBLOB, "Also"))
	    *bp = B_ALSO;
	else if (!strcmp (enableBLOB, "Only"))
	    *bp = B_ONLY;
	else
	    *bp = B_NEVER;
}

/* print key attributes and values of the given xml to stderr.
 */
static void
traceMsg (XMLEle *root)
{
	static const char *prtags[] = {
	    "defNumber", "oneNumber",
	    "defText",   "oneText",
	    "defSwitch", "oneSwitch",
	    "defLight",  "oneLight",
	};
	XMLEle *e;
	unsigned int i;

	/* print tag header */
	logMessage ("%s %s %s %s %s %s %s\n", tagXMLEle(root),
						findXMLAttValu(root,"device"),
						findXMLAttValu(root,"name"),
						findXMLAttValu(root,"state"),
						pcdataXMLEle (root),
						findXMLAttValu(root,"perm"),
						findXMLAttValu(root,"message"));

	/* print each array value except BLOBs */
	for (e = nextXMLEle(root,1); e; e = nextXMLEle(root,0))
	    for (i = 0; i < sizeof(prtags)/sizeof(prtags[0]); i++)
		if (strcmp (prtags[i], tagXMLEle(e)) == 0)
		    logMessage ("%10s='%s'\n", findXMLAttValu(e,"name"),
							    pcdataXMLEle (e));
}

/* fill s with current UT string, assumed to be "long enough".
 */
static char *
tstamp (char *s)
{
	struct timeval tv;
	struct tm *tp;
	time_t t;
	int n;

	gettimeofday (&tv, NULL);
	t = (time_t) tv.tv_sec;
	tp = gmtime (&t);
	n = strftime (s, 100, "%Y-%m-%dT%H:%M:%S", tp);  /* 100 ?! */
	sprintf (s+n, ".%03ld", tv.tv_usec/1000L);
	return (s);
}

/* log any message in root (known to be from device dev)
 */
static void
logDvrMsg (XMLEle *root, char *dev)
{
	char stamp[64];
	char *ts, *ms;

	/* get message, if any */
	ms = findXMLAttValu (root, "message");
	if (!ms[0])
	    return;

	/* lock access to log file */
	pthread_mutex_lock (&log_lock);

	/* get timestamp now if not provided */
	ts = findXMLAttValu (root, "timestamp");
	if (!ts[0])
	    ts = tstamp (stamp);

	/* append to ldir, or stderr */
	if (ldir) {
	    char logfn[1024];
	    FILE *fp;
	    snprintf (logfn, sizeof(logfn), "%s/%.10s.islog", ldir, ts);
	    fp = fopen (logfn, "a");
	    if (fp) {
		fprintf (fp, "%s: %s: %s\n", ts, dev, ms);
		fclose (fp);
	    } else {
                fprintf (stderr, "%s: %s\n", logfn, strerror(errno));
                exit(1);
            }
	} else
	    fprintf (stderr, "%s: %s: %s\n", ts, dev, ms);

	/* release log file */
	pthread_mutex_unlock (&log_lock);
}

/* log internal error or trace message.
 * marshal into linear order via log_lock.
 */
static void
logMessage (const char *fmt, ...)
{
	char ts[64];
	va_list ap;

	/* lock access to log file */
	pthread_mutex_lock (&log_lock);

	(void) tstamp (ts);
	va_start (ap, fmt);

	/* append to ldir, or stderr */
	if (ldir) {
	    char logfn[1024];
	    FILE *fp;
	    snprintf (logfn,sizeof(logfn), "%s/%.10s.islog", ldir, ts);
	    fp = fopen (logfn, "a");
	    if (fp) {
		fprintf (fp, "%s: ", ts);
		vfprintf (fp, fmt, ap);
		fclose (fp);
	    } else {
                fprintf (stderr, "%s: %s\n", logfn, strerror(errno));
                exit(1);
            }
	} else {
	    fprintf (stderr, "%s: ", ts);
	    vfprintf (stderr, fmt, ap);
	}

	va_end (ap);

	/* release log file */
	pthread_mutex_unlock (&log_lock);
}

/* exactly like strncpy() but insures dst is terminated if src happens to have exactly n chars.
 */
static char *
strncpyz (char *dst, const char *src, int n)
{
	char *dp = strncpy (dst, src, n);
	dp[n] = '\0';
	return (dp);
}

/* sleep that uses select(2), just to avoid the signals sleep(3) might use.
 */
static void
ssleep (int ms)
{
	struct timeval tv;

	tv.tv_sec = ms/1000;
	tv.tv_usec = (ms%1000)*1000;
	select (0, NULL, NULL, NULL, &tv);
}

/* fatal error: log and abort */
static void
Bye (const char *fmt, ...)
{
	char msg[1024];
	va_list ap;

	va_start (ap, fmt);
	vsnprintf (msg, sizeof(msg), fmt, ap);
	va_end (ap);

	logMessage (msg);

	abort();
}
