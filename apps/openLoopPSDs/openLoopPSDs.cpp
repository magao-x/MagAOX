/** \file openLoopPSDs.cpp
  * \brief The MagAO-X openLoopPSDs main program source file.
  *
  * \ingroup openLoopPSDs_files
  */

#include "openLoopPSDs.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::openLoopPSDs xapp;

   return xapp.main(argc, argv);

}
