/** \file pupilAlign.cpp
  * \brief The MagAO-X Pyramid Pupil Alignment main-program sourcefile.
  *
  * \ingroup pupilAlign_files
  */

#include "pupilAlign.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::pupilAlign xapp;

   return xapp.main(argc, argv);

}
