/** \file closedLoopIndi.cpp
  * \brief The MagAO-X INDI Closed Loop main program source file.
  *
  * \ingroup closedLoopIndi_files
  */

#include "closedLoopIndi.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::closedLoopIndi xapp;

   return xapp.main(argc, argv);

}
