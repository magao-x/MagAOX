/** \file dmRecon.cpp
  * \brief The MagAO-X DM shape reconstructor
  *
  * \ingroup dmRecon_files
  */

#include "dmRecon.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::dmRecon xapp;

   return xapp.main(argc, argv);

}
