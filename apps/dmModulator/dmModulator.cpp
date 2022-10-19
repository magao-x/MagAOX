/** \file dmModulator.cpp
  * \brief The MagAO-X DM modulator main program source file.
  *
  * \ingroup dmModulator_files
  */

#include "dmModulator.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::dmModulator xapp;

   return xapp.main(argc, argv);

}
