/* just like connect(2) but tries to time out after TOUT.
 * return 0 if ok, else -1.
 */


#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int tout (int to_ms, int fd);

int
connect_to (int sockfd, struct sockaddr *serv_addr, int addrlen, int to_ms)
{
        unsigned int len;
        int err;
        int flags;
        int ret;

        /* set socket non-blocking */
        flags = fcntl (sockfd, F_GETFL, 0);
        (void) fcntl (sockfd, F_SETFL, flags | O_NONBLOCK);

        /* start the connect */
        ret = connect (sockfd, serv_addr, addrlen);
        if (ret < 0 && errno != EINPROGRESS)
            return (-1);

        /* wait for sockfd to become useable */
        ret = tout (to_ms, sockfd);
        if (ret < 0)
            return (-1);

        /* verify connection really completed */
        len = sizeof(err);
        err = 0;
        ret = getsockopt (sockfd, SOL_SOCKET, SO_ERROR, (char *) &err, &len);
        if (ret < 0)
            return (-1);
        if (err != 0) {
            errno = err;
            return (-1);
        }

        /* looks good - restore blocking */
        (void) fcntl (sockfd, F_SETFL, flags);
        return (0);
}

static int
tout (int to_ms, int fd)
{
        fd_set rset, wset;
        struct timeval tv;
        int ret;

        FD_ZERO (&rset);
        FD_ZERO (&wset);
        FD_SET (fd, &rset);
        FD_SET (fd, &wset);

        tv.tv_sec = to_ms/1000;
        tv.tv_usec = to_ms%1000;

        ret = select (fd+1, &rset, &wset, NULL, &tv);
        if (ret > 0)
            return (0);
        if (ret == 0)
            errno = ETIMEDOUT;
        return (-1);
}


