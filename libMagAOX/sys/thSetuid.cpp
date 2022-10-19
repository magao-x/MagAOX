/** \file thSetuid.cpp 
  * \brief Set euid in a single thread
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup sys_files
  */

#include "thSetuid.hpp"


#include <sys/syscall.h>

namespace MagAOX 
{
namespace sys 
{

int th_seteuid(uid_t euid /**< [in] the desired new effective user id */)
{
   int rv = syscall(SYS_setreuid, -1, euid);
   
   return rv;
}

  


} //namespace sys 
} //namespace MagAOX 

 
