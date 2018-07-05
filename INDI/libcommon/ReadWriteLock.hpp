/// ReadWriteLock.hpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_READ_WRITE_LOCK_HPP
#define PCF_READ_WRITE_LOCK_HPP

#include <stdexcept>
#include <pthread.h>
#include <string.h>

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class ReadWriteLock
{
  public:
    // This is a wrapper class for a read lock which will unlock when
    // the class goes out of scope.
    // todo: migrate it to its own file.
    class AutoRLock
    {
      private:
        // This is private so the class may not be instantiated this way.
        AutoRLock()
        {
          m_rwlock = NULL;
        }
      public:
        AutoRLock( ReadWriteLock *rwlock )
        {
          m_rwlock = rwlock;
          if ( m_rwlock == NULL )
            throw std::invalid_argument( "Read-write lock pointer is NULL" );
          m_rwlock->lockRead();
        }
        ~AutoRLock()
        {
          if ( m_rwlock != NULL )
            m_rwlock->unlockRead();
        }
      private:
        ReadWriteLock *m_rwlock;
    };
    // This is a wrapper class for a write lock which will unlock when
    // the class goes out of scope.
    // todo: migrate it to its own file.
    class AutoWLock
    {
      private:
        // This is private so the class may not be instantiated this way.
        AutoWLock()
        {
          m_rwlock = NULL;
        }
      public:
        AutoWLock( ReadWriteLock *rwlock )
        {
          m_rwlock = rwlock;
          if ( m_rwlock == NULL )
            throw std::invalid_argument( "Read-write lock pointer is NULL" );
          m_rwlock->lockWrite();
        }
        ~AutoWLock()
        {
          if ( m_rwlock != NULL )
            m_rwlock->unlockWrite();
        }
      private:
        ReadWriteLock *m_rwlock;
    };

    // Constructor/destructor/operators.
  public:
    ReadWriteLock()
    {
      int nErr = 0;
      if ( ( nErr = pthread_rwlock_init( &m_idLock, NULL ) ) != 0 )
        throw std::runtime_error( std::string( "ReadWriteLock: " ) + strerror( nErr ) );
    }
    virtual ~ReadWriteLock() noexcept(false)
    {
      int nErr = 0;
      if ( ( nErr = pthread_rwlock_destroy( &m_idLock ) ) != 0 )
        throw std::runtime_error( std::string( "ReadWriteLock: " ) + strerror( nErr ) );
    }

  private:
    ReadWriteLock( const ReadWriteLock & )
    {
    }
    const ReadWriteLock &operator =( const ReadWriteLock & )
    {
      return *this;
    }

    // Methods.
  public:
    void lockRead()
    {
      int nErr = 0;
      if ( ( nErr = pthread_rwlock_rdlock( &m_idLock ) ) != 0 )
        throw std::runtime_error( std::string( "lockRead: " ) + strerror( nErr ) );
    }
    void lockWrite()
    {
      int nErr = 0;
      if ( ( nErr = pthread_rwlock_wrlock( &m_idLock ) ) != 0 )
        throw std::runtime_error( std::string( "lockWrite: " ) + strerror( nErr ) );
    }
    void unlockRead()
    {
      int nErr = 0;
      if ( ( nErr = pthread_rwlock_unlock( &m_idLock ) ) != 0 )
        throw std::runtime_error( std::string( "unlockRead: " ) + strerror( nErr ) );
    }
    void unlockWrite()
    {
      int nErr = 0;
      if ( ( nErr = pthread_rwlock_unlock( &m_idLock ) ) != 0 )
        throw std::runtime_error( std::string( "unlockWrite: " ) + strerror( nErr ) );
    }

    // Variables.
  private:
    /// id or handle of the ReadWriteLock created.
    pthread_rwlock_t m_idLock;

}; // Class ReadWriteLock
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_READ_WRITE_LOCK_HPP
