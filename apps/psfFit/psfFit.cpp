/** \file psfFit.cpp
  * \brief The MagAO-X Pyramid psfFit Fitter main-program sourcefile.
  *
  * \ingroup psfFitFit_files
  */

#include "psfFit.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::psfFit xapp;

   return xapp.main(argc, argv);

}
