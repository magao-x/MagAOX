/** \file dmPokeCenter.cpp
  * \brief The MagAO-X DM Pokde Centering main program source file.
  *
  * \ingroup dmPokeCenter_files
  */

#include "dmPokeXCorr.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::dmPokeXCorr xapp;

   return xapp.main(argc, argv);

}
