/** \file jitterPSD.cpp
  * \brief Calculates the PSDs of the movement and Strehl ratio of the camtip
  * images
  *
  * \ingroup jitterPSD_files
  */

#include "jitterPSD.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::jitterPSD xapp;

   return xapp.main(argc, argv);

}
