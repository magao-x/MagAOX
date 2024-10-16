/** \file semUtilsDerived.hpp
  * \brief XWC app semaphore Utilities to use CRTP derived classes
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * 
  * \ingroup app_files
  */

#ifndef app_semUtilsDerived_hpp
#define app_semUtilsDerived_hpp

/// Add the wait time to a timespec for a sem_timedwait call, with no value returned on error, using the derived class
/** An error would be generated by clock_gettime.
  * 
  * \param ts is the timespec to modify, should be set to current time
  * \param sec is the number of seconds to add to ts
  * \param nsec is the number of nanoseconds to add to ts
  *  
  */
#define XWC_SEM_WAIT_TS_RETVOID_DERIVED( ts, sec, nsec )                                        \
    if(clock_gettime(CLOCK_REALTIME, &ts) < 0)                                                  \
    {                                                                                           \
        derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); \
        return; /*will trigger a shutdown*/                                                     \
    }                                                                                           \
    ts.tv_sec += sec;                                                                           \
    mx::sys::timespecAddNsec(ts, nsec);

/// Add the wait time to a timespec for a sem_timedwait call, with -1 returned on error.
/** An error would be generated by clock_gettime
  * 
  * \param ts is the timespec to modify, should be set to current time
  * \param sec is the number of seconds to add to ts
  * \param nsec is the number of nanoseconds to add to ts
  *  
  */
#define XWC_SEM_WAIT_TS_DERIVED( ts, sec, nsec )                             \
    if(clock_gettime(CLOCK_REALTIME, &ts) < 0)                               \
    {                                                                        \
        derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); \
        return -1;                                                           \
    }                                                                        \
    ts.tv_sec += sec;                                                        \
    mx::sys::timespecAddNsec(ts, nsec);

/// Perform a sem_timedwait in the context of a standard loop in MagAO-X code using the derived class
/**
  * \param sem the semaphore
  * \param ts the timespec with the time to wait until
  *  
  */
#define XWC_SEM_TIMEDWAIT_LOOP_DERIVED( sem, ts )                                            \
    if(sem_timedwait(&sem, &ts) != 0)                                                        \
    {                                                                                        \
        /* Check for why we timed out */                                                     \
        /* EINTER probably indicates time to shutdown, loop wil exit if m_shutdown is set */ \
        /* ETIMEDOUT just means keep waiting */                                              \
        if(errno == EINTR || errno == ETIMEDOUT) continue;                                   \
                                                                                             \
        /*Otherwise, report an error.*/                                                      \
        derivedT::template log<software_error>({__FILE__, __LINE__,errno, "sem_timedwait"}); \
        break;                                                                               \
    }

#define XWC_SEM_FLUSH_DERIVED( sem )                                                          \
{                                                                                     \
    int semval;                                                                       \
    if(sem_getvalue( &sem, &semval)<0)                                                \
    {                                                                                 \
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__,errno, "sem_getvalue"});    \
    }                                                                                 \
    for(int i = 0; i < semval; ++i)                                                   \
    {                                                                                 \
        if(sem_trywait(&sem) != 0)                                                    \
        {                                                                             \
            if(errno == EAGAIN) break;                                                \
            return derivedT::template log<software_error,-1>({__FILE__, __LINE__,errno, "sem_trywait"}); \
        }                                                                             \
    }                                                                                 \
}     


#endif //app_semUtilsDerived_hpp
