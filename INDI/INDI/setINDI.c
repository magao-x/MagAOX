/* connect to an INDI server and set one or more device.property.element.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#include "indiapi.h"
#include "connect_to.h"
#include "lilxml.h"
#include "base64.h"

/* table of INDI definition elements we can set
 * N.B. do not change defs[] order, they are indexed via -x/-n/-s args
 */
typedef struct {
    const char *defType;			/* defXXXVector name */
    const char *defOne;				/* defXXX name */
    const char *newType;			/* newXXXVector name */
    const char *oneType;			/* oneXXX name */
} INDIDef;
static INDIDef defs[] = {
    {"defTextVector",   "defText",   "newTextVector",   "oneText"},
    {"defNumberVector", "defNumber", "newNumberVector", "oneNumber"},
    {"defSwitchVector", "defSwitch", "newSwitchVector", "oneSwitch"},
    {"defBLOBVector",   "defBLOB",   "newBLOBVector",   "oneBLOB"},
};
#define	NDEFS	((int)(sizeof(defs)/sizeof(defs[0])))

#define INDIPORT	7624		/* default port */
static char host_def[] = "localhost";	/* default host name */

static char *me;			/* our name for usage message */
static char *host = host_def;		/* working host name */
static int port = INDIPORT;		/* working port number */
static int verbose;			/* report extra info */
static int directfd = -1;		/* direct filedes to server, if >= 0 */
static FILE *svrrfp, *svrwfp;		/* read and file FILE to server */
#define TIMEOUT         2               /* default timeout, secs */
static int timeout = TIMEOUT;           /* working timeout, secs */
static LilXML *lillp;			/* XML parser context */
static int qflag;			/* don't show some error messages */
static int wflag;			/* whether to wait for Ok or Alert */
static int mflag;			/* show any messages associated with spec properties */

typedef struct {
    char *e, *v;			/* element name and value */
    int ok;				/* set when match with a getProps return */
} SetEV;

typedef struct {
    char *d;				/* device */
    char *p;				/* property */
    SetEV *ev;				/* elements */
    int nev;				/* n elements */
    int state;				/* 0..3 */
    INDIDef *dp;			/* one of defs if known, else NULL */
    char missing[1024];			/* list of any missing elements */
    int sent;				/* set when successfully sent to avoid repeats */
} SetSpec;

static SetSpec *sets;			/* set of properties to set */
static int nsets;


static void usage (void);
static void sendGetProps(void);
static int crackSpec (int *acp, char **avp[]);
static void openINDIServer(void);
static void listenINDI (void);
static int finished (void);
static void onAlarm (int dummy);
static int readServerChar (void);
static void sendSet (XMLEle *root);
static void checkState (XMLEle *root);
static void scanEV (SetSpec *specp, char ev[]);
static void scanEEVV (SetSpec *specp, char *ep, char ev[]);
static void scanEVEV (SetSpec *specp, char ev[]);
static void sendNew (FILE *fp, INDIDef *dp, SetSpec *sp);
static void sendOne (FILE *fp, INDIDef *dp, SetSpec *sp);
static void sendBLOB (FILE *fp, SetEV *ep);
static void sendSpecs(void);
static void bye(int n);

int
main (int ac, char *av[])
{
	int stop = 0;
	int allspeced;

	/* save our name */
	me = av[0];

	/* crack args */
	while (!stop && --ac && **++av == '-') {
	    char *s = *av;
	    while (*++s) {
		switch (*s) {

		case 'd':
		    if (ac < 2) {
			fprintf (stderr, "-d requires open fileno\n");
			usage();
		    }
		    directfd = atoi(*++av);
		    ac--;
		    break;

		case 'h':
		    if (directfd >= 0) {
			fprintf (stderr, "Can not combine -d and -h\n");
			usage();
		    }
		    if (ac < 2) {
			fprintf (stderr, "-h requires host name\n");
			usage();
		    }
		    host = *++av;
		    ac--;
		    break;

		case 'm':	/* show messages */
		    mflag++;
		    wflag++;
		    break;

		case 'p':
		    if (directfd >= 0) {
			fprintf (stderr, "Can not combine -d and -p\n");
			usage();
		    }
		    if (ac < 2) {
			fprintf (stderr, "-p requires tcp port number\n");
			usage();
		    }
		    port = atoi(*++av);
		    ac--;
		    break;

		case 'q':	/* quiet */
		    qflag++;
		    break;

		case 't':
		    if (ac < 2) {
			fprintf (stderr, "-t requires timeout\n");
			usage();
		    }
		    timeout = atoi(*++av);
		    ac--;
		    break;

		case 'v':	/* verbose */
		    verbose++;
		    break;

		case 'w':	/* wait */
		    wflag++;
		    break;

		case 'x':	/* FALLTHRU */
		case 'n':	/* FALLTHRU */
		case 's':	/* FALLTHRU */
		case 'b':
		    /* stop if see one of the property types */
		    stop = 1;
		    break;

		default:
		    fprintf (stderr, "Unknown flag: %c\n", *s);
		    usage();
		}
	    }
	}

	/* now ac args starting at av[0] */
	if (ac < 1)
	    usage();

	/* sanity check */
	if (qflag && verbose) {
	    fprintf (stderr, "-q and -v together are too confusing\n");
	    usage();
	}

	/* crack each property, add to sets[]  */
	allspeced = 1;
	do {
	    if (!crackSpec (&ac, &av))
		allspeced = 0;
	} while (ac > 0);
	if (allspeced && wflag) {
	    fprintf (stderr, "Can not use -w with type spec flags\n");
	    bye(1);
	}

	/* stay here if IO trouble */
	signal (SIGPIPE, SIG_IGN);

	/* open connection */
	if (directfd >= 0) {
	    svrrfp = fdopen (directfd, "r");
	    svrwfp = fdopen (directfd, "w");
	    if (!svrrfp || !svrwfp) {
		fprintf (stderr, "Direct fd %d is not valid\n", directfd);
		exit(1);
	    }
	    setbuf (svrrfp, NULL);		/* don't absorb next guy's stuff */
	    setbuf (svrwfp, NULL);		/* immediate writes */
	    if (verbose)
		fprintf (stderr, "Using direct fd %d\n", directfd);
	} else {
	    openINDIServer();
	    if (verbose)
		fprintf (stderr, "Connected to %s on port %d\n", host, port);
	}

	/* build a parser context for cracking XML responses */
	lillp = newLilXML();

	/* just send if all type-speced, else check with server */
	if (allspeced) {
	    sendSpecs();
	} else {
	    /* issue getProperties */
	    sendGetProps();

	    /* listen for properties, set when see any we recognize */
	    listenINDI();
	}

	return (0);
}

static void
usage()
{
	fprintf(stderr, "Purpose: set one or more writable INDI properties\n");
	fprintf(stderr, "%s\n", "$Revision: 1.12 $");
	fprintf(stderr, "Usage: %s [options] {[type] spec} ...\n", me);
	fprintf(stderr, "\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -d f  : use file descriptor f already open to server\n");
	fprintf(stderr, "  -h h  : alternate host, default is %s\n", host_def);
	fprintf(stderr, "  -m    : show property messages (implies -w)\n");
	fprintf(stderr, "  -p p  : alternate port, default is %d\n", INDIPORT);
	fprintf(stderr, "  -q    : suppress some error messages\n");
	fprintf(stderr, "  -t t  : max time to wait, default is %d secs\n",TIMEOUT);
	fprintf(stderr, "  -v    : verbose (more are cumulative)\n");
	fprintf(stderr, "  -w    : wait for state to be Ok or Alert - can not be used with type flags\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Spec may be either:\n");
	fprintf(stderr, "    device.property.e1[;e2...]=v1[;v2...]\n");
	fprintf(stderr, "  or\n");
	fprintf(stderr, "    device.property.e1=v1[;e2=v2...]\n");
	fprintf(stderr, "    (This form allows embedded ; in v as \\;)\n");
	fprintf(stderr, "  If Property is a BLOB, each v is a file name\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "If spec is preceded by one of the following flags to indicate its type,\n");
	fprintf(stderr, "the transaction is much more efficient but there is no error checking:\n");
	fprintf(stderr, "  -x    : Text\n");
	fprintf(stderr, "  -n    : Number\n");
	fprintf(stderr, "  -s    : Switch\n");
	fprintf(stderr, "  -b    : BLOB\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Exit status:\n");
	fprintf(stderr, "  0: all settings successful\n");
	fprintf(stderr, "  1: at least one setting was invalid\n");
	fprintf(stderr, "  2: real trouble, try repeating with -v\n");

	exit (2);
}

/* send getProperties for each unique device
 */
static void
sendGetProps()
{
	int i, j;

	for (i = 0; i < nsets; i++) {
	    SetSpec *sp = &sets[i];
	    for (j = 0; j < i; j++)
		if (strcmp (sets[j].d, sp->d) == 0)
		    break;
	    if (i == j) {
		if (verbose)
		    fprintf (stderr, "Querying for %s properties\n", sp->d);
		if (fprintf(svrwfp, "<getProperties version='%g' device='%s'/>\n", INDIV, sp->d) < 0) {
		    fprintf (stderr, "Failed sending getProps: %s\n", strerror(errno));
		    bye(1);
		}
	    }
	}
}

/* crack property set spec, add to sets [], move to next spec.
 * return 1 if see a type
 */
static int
crackSpec (int *acp, char **avp[])
{
	char d[128], p[128], ev[2048];
	char *spec = *avp[0];
	INDIDef *dp = NULL;

	/* check if first arg is type indicator */
	if (spec[0] == '-') {
	    switch (spec[1]) {
	    case 'x':	dp = &defs[0]; break;
	    case 'n':	dp = &defs[1]; break;
	    case 's':	dp = &defs[2]; break;
	    case 'b':	dp = &defs[3]; break;
	    default:
		fprintf (stderr, "Bad property type: %s\n", spec);
		usage();
	    }
	    (*acp)--;
	    (*avp)++;
	    spec = *avp[0];
	}

	/* then scan arg for property spec.
	 * N.B. can't use %s for ev because it stops at whitespace; the %[ -~]
	 *   includes all printable ASCII characters from space through squiggle.
	 */
	if (sscanf (spec, "%127[^.].%127[^.].%2047[ -~]", d, p, ev) != 3) {
	    fprintf (stderr, "Malformed property spec: %s\n", spec);
	    usage();
	}

	/* add to list */
	if (!sets)
	    sets = (SetSpec *) malloc (1);		/* seed realloc */

   SetSpec * tmp_sets;
	tmp_sets = (SetSpec *) realloc (sets, (nsets+1)*sizeof(SetSpec));

   if(tmp_sets == NULL)
   {
      //free(sets);
      sets= NULL;
      return 0;
   }
   sets = tmp_sets;
   tmp_sets = NULL;

	memset (&sets[nsets], 0, sizeof(SetSpec));
	sets[nsets].d = strcpy ((char *)malloc(strlen(d)+1), d);
	sets[nsets].p = strcpy ((char *)malloc(strlen(p)+1), p);
	sets[nsets].dp = dp;
	sets[nsets].ev = (SetEV *) malloc (1);		/* seed realloc */
	scanEV (&sets[nsets++], ev);

	/* update caller's pointers */
	(*acp)--;
	(*avp)++;

	/* return 1 if saw a spec */
	return (dp ? 1 : 0);
}

/* open a read and write connection to host and port on svrrfp and svrwfp or die.
 * exit if trouble.
 */
static void
openINDIServer ()
{
	struct sockaddr_in serv_addr;
	struct hostent *hp;
	int sockfd;
        int i;

	/* lookup host address */
	hp = gethostbyname (host);
	if (!hp) {
	    perror(host);
	    exit (2);
	}

	/* create a socket to the INDI server */
	(void) memset ((char *)&serv_addr, 0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr =
			    ((struct in_addr *)(hp->h_addr_list[0]))->s_addr;
	serv_addr.sin_port = htons(port);
	if ((sockfd = socket (AF_INET, SOCK_STREAM, 0)) < 0) {
	    perror ("socket");
	    exit(2);
	}

	/* connect */
        for (i = 0; i < 2; i++)
            if (connect_to (sockfd,(struct sockaddr *)&serv_addr,sizeof(serv_addr), 1000 ) == 0)
                break;
        if (i == 2) {
	    perror ("connect");
	    exit(2);
	}

	/* prepare for line-oriented i/o to client */
	svrrfp = fdopen (sockfd, "r");
	svrwfp = fdopen (sockfd, "w");
	setbuf (svrwfp, NULL);		/* immediate writes */
}

/* listen for property reports, send new sets if match */
static void
listenINDI ()
{
	char msg[1024];

	/* arrange to call onAlarm() if not seeing any more defXXX */
	signal (SIGALRM, onAlarm);
	alarm (timeout);

	/* read from server, exit if find all properties */
	while (1) {
	    XMLEle *root = readXMLEle (lillp, readServerChar(), msg);
	    if (root) {
		/* found a complete XML element */
		if (verbose > 2)
		    prXMLEle (stderr, root, 0);
		sendSet (root);
		if (wflag)
		    checkState (root);
		if (finished() == 0)
		    bye (0);		/* found all we want or saw Alert */
		delXMLEle (root);	/* not yet, delete and continue */
	    } else if (msg[0]) {
		fprintf (stderr, "Bad XML from %s/%d: %s\n", host, port, msg);
		bye(2);
	    }
	}
}

/* return 0 if we are sure we set everything we wanted to or we saw any Alert, else -1.
 */
static int
finished ()
{
	int i, j;

	for (i = 0; i < nsets; i++) {
	    if (sets[i].state == 3)
		return (0);
	    if (sets[i].missing[0])
		return (-1);
	    if (wflag && (sets[i].state == 0 || sets[i].state == 2))
		return (-1);
	    for (j = 0; j < sets[i].nev; j++)
		if (!sets[i].ev[j].ok)
		    return (-1);
	}
	return (0);
}

/* called after timeout seconds because we did not find something we trying
 * to set.
 */
static void
onAlarm (int dummy)
{
	int i, j;

	for (i = 0; i < nsets; i++) {
	    if (sets[i].missing[0])
		if (!qflag)
		    fprintf (stderr, "Extra %s.%s.%s from %s:%d\n", sets[i].d,
				sets[i].p, sets[i].missing, host, port);
	    for (j = 0; j < sets[i].nev; j++) {
		if (!sets[i].ev[j].ok)
		    if (!qflag)
			fprintf (stderr, "No %s.%s.%s from %s:%d\n", sets[i].d,
				    sets[i].p, sets[i].ev[j].e, host, port);
	    }
	}

	if (!qflag)
	    fprintf (stderr, "timed out\n");
	exit (1);
}

static int
readServerChar ()
{
	int c = fgetc (svrrfp);

	if (c == EOF) {
	    if (ferror(svrrfp))
		perror ("read");
	    else
		fprintf (stderr,"INDI server %s:%d disconnected\n", host, port);
	    bye (2);
	}

	return (c);
}

/* This is called on the arrival of each new INDI message in response
 * to a getProprties. The idea is to see whether any of the specs we
 * have been given are an exact match to a def*. If there is a match and we have
 * never sent this spec before then we send the new spec, otherwise we just return.
 * If we never see all of our specs match, we'll time out and onAlarm will
 * report the details.
 */
static void
sendSet (XMLEle *root)
{
	char *rtype, *rdev, *rprop;
	XMLEle *ep;
	int t, s, i;

	/* type must be def* */
	rtype = tagXMLEle (root);
	for (t = 0; t < NDEFS; t++)
	    if (strcmp (rtype, defs[t].defType) == 0)
		break;
	if (t == NDEFS)
	    return;
	alarm (timeout);	/* reset timeout */

	/* check each set for matching device and property name, send if ok and new */
	rdev  = findXMLAttValu (root, "device");
	rprop = findXMLAttValu (root, "name");
	if (verbose > 2)
	    fprintf (stderr, "Read definition for %s.%s\n", rdev, rprop);
	for (s = 0; s < nsets; s++) {
	    SetSpec *sp = &sets[s];

	    if (!strcmp (rdev, sp->d) && !strcmp (rprop, sp->p)) {
		int nok = 0;

		/* found device and name, confirm writable */
		if (!strchr (findXMLAttValu (root, "perm"), 'w')) {
		    fprintf (stderr, "%s.%s is read-only\n", rdev, rprop);
		    bye (1);
		}

		/* reset list of missing elements */
		sp->missing[0] = '\0';

		/* check matching elements */
		for (ep = nextXMLEle(root,1); ep; ep = nextXMLEle(root,0)) {
		    char *tag = tagXMLEle(ep);

		    if (!strcmp(tag, defs[t].defOne)) {
			char *el = findXMLAttValu (ep,"name");
			int found = 0;
			for (i = 0; i < sp->nev; i++) {
			    if (!strcmp(el, sp->ev[i].e)) {
				sp->ev[i].ok = 1;
				nok++;
				if (verbose)
				    fprintf (stderr, "Confirmed %s.%s.%s\n", sp->d, sp->p,
							sp->ev[i].e);
				found++;
				break;
			    }
			}
			if (!found) {
			    if (verbose)
				fprintf (stderr, "Reported %s.%s.%s but not in spec\n",
				    sp->d, sp->p, el);
			    strcat (sp->missing, el);
			    strcat (sp->missing, ",");
			}
		    }
		}
		if (sp->missing[0])
		    return;	/* elements are in root not in this spec */
		if (nok != sp->nev)
		    return;	/* elements are in this spec not in root */

		/* all element names found, send new values */
		sendNew (svrwfp, &defs[t], sp);
	    }
	}
}

/* send the given set specification of the given INDI type to channel on fp if not
 * already sent as indicated by sp->sent.
 */
static void
sendNew (FILE *fp, INDIDef *dp, SetSpec *sp)
{
	if (sp->sent)
	    return;

	sendOne (fp, dp, sp);

	if (verbose > 1)
	    sendOne (stderr, dp, sp);

	sp->sent = 1;
}

static void
sendOne (FILE *fp, INDIDef *dp, SetSpec *sp)
{
	int i;

	/* send INDI syntax to fp */
	fprintf (fp, "<%s device='%s' name='%s'>\n", dp->newType, sp->d, sp->p);
	for (i = 0; i < sp->nev; i++) {
	    if (strcmp (dp->oneType, "oneBLOB") == 0)
		sendBLOB (fp, &sp->ev[i]);
	    else
		fprintf (fp, "  <%s name='%s'>%s</%s>\n", dp->oneType,
			sp->ev[i].e, entityXML(sp->ev[i].v), dp->oneType);
	}

	fprintf (fp, "</%s>\n", dp->newType);
	fflush (fp);
	if (feof(fp) || ferror(fp)) {
	    fprintf (stderr, "Send error: %s\n", strerror(errno));
	    bye(2);
	}
}

/* send one BLOB defined by SetEV */
static void
sendBLOB (FILE *fp, SetEV *ep)
{
	struct stat s;
	int fd, bloblen, l64, i;
	char *dot;
	unsigned char *blob, *encblob;

	/* get file size */
	if (stat (ep->v, &s) < 0) {
	    fprintf (stderr, "Could not get size of %s: %s\n", ep->v,
	    			strerror(errno));
	    bye(1);
	}
	bloblen = s.st_size;

	/* get extension for use as type */
	dot = strchr (ep->v, '.');
	if (!dot) {
	    fprintf (stderr, "No extension found for %s\n", ep->v);
	    bye (1);
	}

	/* open file and convert to base64 */
	fd = open (ep->v, O_RDONLY);
	if (fd < 0) {
	    fprintf (stderr, "Could not open %s: %s\n", ep->v,
	    			strerror(errno));
	    bye(1);
	}
	blob = (unsigned char *) malloc (bloblen);
	i = read (fd, blob, bloblen);
	if (i != bloblen) {
	    if (i == 0)
		fprintf (stderr, "Premature EOF reading %s\n", ep->v);
	    else if (i < 0)
		fprintf (stderr, "Error while reading %s: %s\n", ep->v,
			    strerror(errno));
	    else
		fprintf (stderr, "BLOB file %s is short: %d < %d\n", ep->v,
				i, bloblen);
	    bye(1);
	}
	close (fd);

	/* send message */
	fprintf (fp, "  <oneBLOB\n");
	fprintf (fp, "    name='%s'\n", ep->e);
	fprintf (fp, "    size='%d'\n", bloblen);
	fprintf (fp, "    format='%s'>\n", dot+1);

	encblob = (unsigned char *) malloc (4*bloblen/3+4);
	l64 = to64frombits(encblob, blob, bloblen);
	for (i = 0; i < l64; i += 72)
	    fprintf (fp, "%.72s\n", encblob+i);

	fprintf (fp, "  </oneBLOB>\n");

	free (encblob);
	free (blob);
}

/* called with each incoming message to check whether it contains a set* that
 * matches any spec. If so, record the state and show any Alert messages.
 */
static void
checkState (XMLEle *root)
{
	char *rtype, *rdev, *rprop;
	int s;

	/* type must be set* */
	rtype = tagXMLEle (root);
	if (strncmp (rtype, "set", 3))
	    return;

	/* check  for matching device and property name */
	rdev  = findXMLAttValu (root, "device");
	rprop = findXMLAttValu (root, "name");
	if (verbose > 2)
	    fprintf (stderr, "Read %s for %s.%s\n", rtype, rdev, rprop);
	for (s = 0; s < nsets; s++) {
	    if (!strcmp (rdev, sets[s].d) && !strcmp (rprop, sets[s].p)) {
		char *state = findXMLAttValu (root, "state");

		if (verbose)
		    fprintf (stderr, "%s.%s state %s\n", sets[s].d, sets[s].p, state);
		if (strcmp (state, "Ok") == 0)
		    sets[s].state = 1;
		else if (strcmp (state, "Busy") == 0)
		    sets[s].state = 2;
		else if (strcmp (state, "Alert") == 0)
		    sets[s].state = 3;

		char *msg = findXMLAttValu (root, "message");
		if (msg && msg[0] && ((!qflag && sets[s].state == 3) || mflag))
		    fprintf (stderr, "%s\n", msg);
	    }
	}
}


/* scan ev for element definitions in either of two forms and add to sp:
 *    e1[;e2...]=v1[;v2...]
 *  or
 *    e1=v1[;e2=v2...]
 * exit if nothing sensible found.
 */
static void
scanEV (SetSpec *specp, char ev[])
{
	char *ep, *sp;		/* pointers to = and ; */

	if (verbose > 1)
	    fprintf (stderr, "Scanning assignments %s\n", ev);

	ep = strchr (ev, '=');
	sp = strchr (ev, ';');

	if (!ep) {
	    fprintf (stderr, "Malformed assignment: %s\n", ev);
	    usage();
	}

	if (sp < ep)
	    scanEEVV (specp, ep, ev);	/* including just one E=V */
	else
	    scanEVEV (specp, ev);
}

/* add specs of the form e1[;e2...]=v1[;v2...] to sp.
 * v is pointer to equal sign.
 * exit if trouble.
 * N.B. e[] and v[] are modified in place.
 */
static void
scanEEVV (SetSpec *sp, char *v, char *e)
{
	static char sep[] = ";";
	char *ec, *vc;

	*v++ = '\0';

	while (1) {
	    char *e0 = strtok_r (e, sep, &ec);
	    char *v0 = strtok_r (v, sep, &vc);

	    if (!e0 && !v0)
		break;
	    if (!e0) {
		fprintf (stderr, "More values than elements for %s.%s\n", sp->d, sp->p);
		bye(2);
	    }
	    if (!v0) {
		fprintf (stderr, "More elements than values for %s.%s\n", sp->d, sp->p);
		bye(2);
	    }

	    sp->ev = (SetEV *) realloc (sp->ev, (sp->nev+1)*sizeof(SetEV));
	    sp->ev[sp->nev].e = strcpy ((char *)malloc(strlen(e0)+1), e0);
	    sp->ev[sp->nev].v = strcpy ((char *)malloc(strlen(v0)+1), v0);
	    sp->ev[sp->nev].ok = 0;
	    if (verbose > 1)
		fprintf (stderr, "Found assignment %s=%s\n", sp->ev[sp->nev].e, sp->ev[sp->nev].v);
	    sp->nev++;

	    e = NULL;
	    v = NULL;
	}
}

/* add specs of the form e1=v1[;e2=v2...] to sp.
 * exit if trouble.
 * N.B. ev[] is modified in place.
 */
static void
scanEVEV (SetSpec *sp, char ev[])
{
	char *s, *v;
	int last = 0;

	do {
	    /* look for next ; not preceded by \ and replace any such \ with ' '
	     */
	    s = ev;
	    while ((s = strchr(s, ';')) && s[-1] == '\\') {
		s[-1] = ' ';
		s++;
	    }
	    if (s)
		*s++ = '\0';
	    else
		last = 1;

	    v = strchr (ev, '=');
	    if (v)
		*v++ = '\0';
	    else {
		fprintf (stderr, "Malformed assignment: %s\n", ev);
		usage();
	    }

	    sp->ev = (SetEV *) realloc (sp->ev, (sp->nev+1)*sizeof(SetEV));
	    sp->ev[sp->nev].e = strcpy ((char *)malloc(strlen(ev)+1), ev);
	    sp->ev[sp->nev].v = strcpy ((char *)malloc(strlen(v)+1), v);
	    sp->ev[sp->nev].ok = 0;
	    if (verbose > 1)
		fprintf (stderr, "Found assignment %s=%s\n", sp->ev[sp->nev].e, sp->ev[sp->nev].v);
	    sp->nev++;

	    ev = s;

	} while (!last);
}

/* send each SetSpec, all of which have a known type, to svrwfp
 */
static void
sendSpecs()
{
	int i;

	for (i = 0; i < nsets; i++) {
	    sendNew (svrwfp, sets[i].dp, &sets[i]);
	    if (verbose)
		sendNew (stderr, sets[i].dp, &sets[i]);
	}
}

/* cleanly close svr/wfp then exit(n)
 */
static void
bye(int n)
{
	if ((svrwfp || svrrfp) && directfd < 0) {
	    int rfd = svrrfp ? fileno(svrrfp) : -1;
	    int wfd = svrwfp ? fileno(svrwfp) : -1;
	    if (rfd >= 0) {
		shutdown (rfd, SHUT_RDWR);
		fclose (svrrfp);	/* also closes rfd */
	    }
	    if (wfd >= 0 && wfd != rfd) {
		shutdown (wfd, SHUT_RDWR);
		fclose (svrwfp);	/* also closes wfd */
	    }
	}

	exit (n);
}
