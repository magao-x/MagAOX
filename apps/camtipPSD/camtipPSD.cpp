/** \file camtipPSD.cpp
  * \brief Calculates the PSDs of the movement and Strehl ratio of the camtip
  * images
  *
  * \ingroup camtipPSD_files
  */

#include "camtipPSD.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::camtipPSD xapp;

   return xapp.main(argc, argv);

}
