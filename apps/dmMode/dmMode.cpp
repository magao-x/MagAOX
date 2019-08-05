/** \file dmMode.cpp
  * \brief The MagAO-X DM mode command main program source file.
  *
  * \ingroup dmMode_files
  */

#include "dmMode.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::dmMode xapp;

   return xapp.main(argc, argv);

}
