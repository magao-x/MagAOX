/** \file kTracker.cpp
  * \brief The MagAO-X K-mirror rotation tracker main program source file.
  *
  * \ingroup kTracker_files
  */

#include "kTracker.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::kTracker xapp;

   return xapp.main(argc, argv);

}
