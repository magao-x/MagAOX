/// MutexLock.hpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_MUTEX_LOCK_HPP
#define PCF_MUTEX_LOCK_HPP

#include <stdexcept>
#include <pthread.h>
#include <string.h>

//#include "../common/Logger.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class MutexLock
{
  public:
    // This is a wrapper class for a mutex lock which will unlock when
    // the class goes out of scope.
    // todo: migrate it to its own file.
    class AutoLock
    {
      private:
        // This is private so the class may not be instantiated this way.
        AutoLock() : m_mutex( NULL )
        {
        }
      public:
        AutoLock( MutexLock *mutex ) : m_mutex( mutex )
        {
          if ( m_mutex == NULL )
            throw std::invalid_argument( "MutexLock::AutoLock: Mutex pointer is NULL" );
          m_mutex->lock();
        }
        ~AutoLock()
        {
          if ( m_mutex != NULL )
            m_mutex->unlock();
        }
      private:
        MutexLock *m_mutex;
    };

    // Constructor/destructor/operators.
  public:
    MutexLock()
    {
      int nErr = 0;
      if ( ( nErr = pthread_mutex_init( &m_idLock, NULL ) ) != 0 )
        throw std::runtime_error( std::string( "MutexLock::MutexLock: " ) + strerror( nErr ) );
      //m_logMsg.clear();
      //m_logMsg << pcf::Logger::enumDebug << "pthread_mutex_init" << std::endl;
    }
    virtual ~MutexLock() noexcept(false)
    {
      int nErr = 0;
      if ( ( nErr = pthread_mutex_destroy( &m_idLock ) ) != 0 )
       return;
      //throw std::runtime_error( std::string( "MutexLock::~MutexLock: " ) + strerror( nErr ) );
      //m_logMsg.clear();
      //m_logMsg << pcf::Logger::enumDebug << "pthread_mutex_destroy" << std::endl;
    }

  private:
    MutexLock( const MutexLock & )
    {
    }
    const MutexLock &operator=( const MutexLock & )
    {
      return *this;
    }

    // Methods.
  public:
    void lock()
    {
      int nErr = 0;
      if ( ( nErr = pthread_mutex_lock( &m_idLock ) ) != 0 )
        throw std::runtime_error( std::string( "MutexLock::lock: " ) + strerror( nErr ) );
      //m_logMsg.clear();
      //m_logMsg << pcf::Logger::enumDebug << "pthread_mutex_lock" << std::endl;
    }
    void unlock()
    {
      int nErr = 0;
      if ( ( nErr = pthread_mutex_unlock( &m_idLock ) ) != 0 )
        throw std::runtime_error( std::string( "MutexLock::unlock: " ) + strerror( nErr ) );
      //m_logMsg.clear();
      //m_logMsg << pcf::Logger::enumDebug << "pthread_mutex_unlock" << std::endl;
    }

    // Variables.
  private:
    /// id or handle of the mutex lock created.
    pthread_mutex_t m_idLock;
    //pcf::Logger m_logMsg;

}; // Class MutexLock
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_MUTEX_LOCK_HPP
