/** \file logdump.cpp
  * \brief A simple utility to dump MagAO-X binary logs to stdout.
  */

#include "logdump.hpp"


/** \todo document this
  * \todo add config for colors, both on/off and options to change.
  */
int main(int argc, char **argv)
{
   logdump ld;

   return ld.main(argc, argv);

}
