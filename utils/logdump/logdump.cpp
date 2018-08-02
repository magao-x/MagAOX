/** \file logdump.cpp
  * \brief A simple utility to dump MagAO-X binary logs to stdout.
  */

#include "logdump.hpp"


/** \todo document this
  * \todo add filters for loglevel, and type.
  * \todo add config option for sleep time.
  * \todo add config for colors, both on/off and options to change.
  */
int main(int argc, char **argv)
{
   logdump ld;

   return ld.main(argc, argv);

}
