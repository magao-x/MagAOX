/* use lilxml to read stdin (or named file). print and/or exit whether the parsing was successful.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#include "lilxml.h"

static int qflag;
static int pflag;

static void usage (char *me);
static XMLEle *parseXML (FILE *fp, const char *filename);

int
main (int ac, char *av[])
{
	char *me = av[0];
	char *filename = NULL;
	FILE *fp;
	XMLEle *root;
	int secn;

        /* crack args */
        while ((--ac > 0) && ((*++av)[0] == '-')) {
            char *s;
            for (s = av[0]+1; *s != '\0'; s++) {
                switch (*s) {
		case 'i':
		    if (ac < 2) {
			fprintf (stderr, "-i requires input file name\n");
			usage (me);
		    }
		    filename = *++av;
		    ac--;
		    break;
                case 'p':
		    pflag++;
		    break;
                case 'q':
		    qflag++;
		    break;
		default:
		    fprintf (stderr, "Unknown flag: %c\n", *s);
		    usage(me);
		    break;
		}
	    }
	}
	if (ac > 0) {
	    fprintf (stderr, "Unexpected extra arguments\n");
	    usage(me);
	}

	/* open filename else use stdin */
	if (filename) {
	    fp = fopen (filename, "r");
	    if (!fp) {
		fprintf (stderr, "%s: %s\n", filename, strerror(errno));
		exit(1);
	    }
	} else {
	    fp = stdin;
	    filename = (char *)"stdin";
	}

	/* keep checking more bits until EOF */
	secn = 1;
	do {
	    if (!qflag)
		fprintf (stderr, "Parsing section %d\n", secn++);
	    root = parseXML (fp, filename);
	    if (root) {
		if (pflag)
		    prXMLEle (stdout, root, 0);
		delXMLEle (root);
	    }
	} while (root);

	/* ok */
	if (!qflag)
	    fprintf (stderr, "No errors found\n");
	return (0);
}

/* print usage and exit
 */
static void
usage (char *me) 
{
	char *rslash;

	/* basename */
	rslash = strrchr (me, '/');
	if (rslash)
	    me = rslash + 1;

	fprintf (stderr, "Purpose: check for valid xml syntax\n");
	fprintf (stderr, "Usage: %s [options]\n", me);
	fprintf (stderr, "  -i f: read file f, else stdin\n");
	fprintf (stderr, "  -q  : quiet, just exit 0/1 whether xml is valid\n");
	fprintf (stderr, "  -p  : print parsed xml on stdout\n");

	exit(1);
}

/* parse more of the given FILE stream.
 * return NULL if EOF, exit(1) if parsing error, XMLEle* if successful.
 */
static XMLEle *
parseXML (FILE *fp, const char *filename)
{
	LilXML *lp = newLilXML();
	char ynot[1024];
	int c;

	while ((c = fgetc(fp)) != EOF) {
	    XMLEle *root = readXMLEle (lp, c, ynot);
	    if (root) {
		delLilXML (lp);
		return (root);
	    } else if (ynot[0]) {
		if (!qflag)
		    fprintf (stderr, "%s: %s\n", filename, ynot);
		exit(1);
	    }
	}

	delLilXML (lp);
	return (NULL);
}
