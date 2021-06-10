/** \file t2wOffloader.cpp
  * \brief The MagAO-X Tweeter To Woofer Offloader
  *
  * \ingroup t2wOffloader_files
  */

#include "lowfsPredCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::lowfsPredCtrl xapp;

   return xapp.main(argc, argv);

}
