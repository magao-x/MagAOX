/** \file logStdFormat.hpp
  * \brief Standard formating of log entries for readable output.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup flatlogs_files
  * 
  * History:
  * - 2017-12-24 created by JRM
  * - 2018-08-18 moved to flatlogs
  */

#ifndef flatlogs_logStdFormat_hpp
#define flatlogs_logStdFormat_hpp

#include "logHeader.hpp"
#include "logPriority.hpp"

namespace flatlogs
{

/// Worker function that formats a log into the standard text representation.
/** \todo change to using a std::ios as input instead of only using std::cout
  *
  * \ingroup logformat
  */
template<typename logT>
void stdFormat( bufferPtrT & logBuffer /**< [in] the binary log buffer */)
{
   logPrioT prio;
   eventCodeT ec;
   timespecX ts;
   msgLenT len;

   logHeader::extractBasicLog( prio, ec, ts, len, logBuffer);

   std::cout << ts.ISO8601DateTimeStrX() << " " << priorityString(prio) << " " << logT::msgString(logHeader::messageBuffer(logBuffer) , len);
}

} //namespace flatlogs

#endif //flatlogs_logStdFormat_hpp
