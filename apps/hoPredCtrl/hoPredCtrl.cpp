/** \file hoPredCtrl.cpp
  * \brief The MagAO-X Tweeter To Woofer Offloader
  *
  * \ingroup hoPredCtrl_files
  */

#include "hoPredCtrl.cuh"


int main(int argc, char **argv)
{
   MagAOX::app::hoPredCtrl xapp;

   return xapp.main(argc, argv);

}
