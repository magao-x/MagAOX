/** \file logdump.cpp
  * \brief A simple utility to dump MagAO-X binary logs to stdout.
  * 
  * \ingroup logdump_files
  */

#include "logdump.hpp"



int main(int argc, char **argv)
{
   logdump ld;

   return ld.main(argc, argv);

}
