/** \file pupilFit.cpp
  * \brief The MagAO-X Pyramid Pupil Fitter main-program sourcefile.
  *
  * \ingroup pupilFit_files
  */

#include "pupilFit.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::pupilFit xapp;

   return xapp.main(argc, argv);

}
