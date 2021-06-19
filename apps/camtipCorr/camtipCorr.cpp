/** \file camtipCorr.cpp
  * \brief The MagAO-X Image Registrator and Strehl
  * ratio monitor
  *
  * \ingroup camtipCorr_files
  */

#include "camtipCorr.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::camtipCorr xapp;

   return xapp.main(argc, argv);

}
