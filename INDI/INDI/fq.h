/* these functions interface to a generic fifo queue.
 */

/* anonymous type, serves as a handle to each queue instance */
typedef struct _FQ FQ;

/* create a new empty queue, with hint about n items to grow when full */
extern FQ *newFQ(int grow);

/* delete a queue and all its contents */
extern void delFQ (FQ *q);

/* add an item to a queue */
extern void pushFQ (FQ *q, void *e);

/* pop the oldest item from a queue */
extern void *popFQ (FQ *q);

/* see but do not pop the oldest item on a queue */
extern void *peekFQ (FQ *q);

/* see the ith element from the head of the queue */
extern void *peekiFQ (FQ *q, int i);

/* return the number of items on a queue */
extern int nFQ (FQ *q);

/* set alternate memory operators (default are stdlib malloc/realloc/free) */
extern void setMemFuncsFQ (void *(*newmalloc)(size_t size),
   void *(*newrealloc)(void *ptr, size_t size),
   void (*newfree)(void *ptr));

/* For RCS Only -- Do Not Edit
 * @(#) $RCSfile: fq.h,v $ $Date: 2007/07/01 19:23:58 $ $Revision: 1.2 $ $Name:  $
 */
