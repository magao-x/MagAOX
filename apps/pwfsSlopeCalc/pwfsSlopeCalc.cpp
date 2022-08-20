/** \file pwfsSlopeCalc.cpp
  * \brief The MagAO-X PWFS Slope Calculator main program source file.
  *
  * \ingroup pwfsSlopeCalc_files
  */

#include "pwfsSlopeCalc.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::pwfsSlopeCalc xapp;

   return xapp.main(argc, argv);

}
