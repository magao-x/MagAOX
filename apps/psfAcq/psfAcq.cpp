/** \file psfAcq.cpp
  * \brief The MagAO-X Pyramid psfAcq Fitter main-program sourcefile.
  *
  * \ingroup psfAcqAcq_files
  */

#include "psfAcq.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::psfAcq xapp;

   return xapp.main(argc, argv);

}

