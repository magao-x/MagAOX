/** \file logstream.cpp
  * \brief A simple utility to dump MagAO-X binary logs to stdout.
  */

#include "logstream.hpp"


/** \todo document this
  * \todo add config for colors, both on/off and options to change.
  */
int main(int argc, char **argv)
{
   logstream ls;

   std::set<std::string> appNames;
   return ls.getAppsWithLogs( appNames );
   
   return 0;

}
