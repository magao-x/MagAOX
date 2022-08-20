/** \file thSetuid.hpp 
  * \brief Set euid in a single thread
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup sys_files
  * History:
  * - 2019-07-12 created by JRM
  */

#ifndef sys_thSetuid_hpp
#define sys_thSetuid_hpp

#include <unistd.h>

namespace MagAOX 
{
namespace sys 
{

/// Sets the effective user id of the calling thread, rather than the whole process
/** Uses the syscall directly so that only the calling thread has modified privileges.
 * 
 *  Ref: http://man7.org/linux/man-pages/man2/seteuid.2.html
 * 
 * \returns 0 on success
 * \returns -1 on error, and errno is set 
 * 
 * \ingroup sys
 */
int th_seteuid(uid_t euid /**< [in] the desired new effective user id */);

} //namespace sys 
} //namespace MagAOX 

#endif //sys_thSetuid_hpp
 
