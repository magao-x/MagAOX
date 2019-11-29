/** \file adcTracker.cpp
  * \brief The MagAO-X ADC Tracker main program source file.
  *
  * \ingroup adcTracker_files
  */

#include "adcTracker.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::adcTracker xapp;

   return xapp.main(argc, argv);

}
