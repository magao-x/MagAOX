/** \file logsurgeon.cpp
  * \brief A simple utility to dump MagAO-X binary logs to stdout.
  * 
  * \ingroup logsurgeon_files
  */

#include "logsurgeon.hpp"



int main(int argc, char **argv)
{
   logsurgeon ls;

   return ls.main(argc, argv);

}
