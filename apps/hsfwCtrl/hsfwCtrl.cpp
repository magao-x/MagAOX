/** \file hsfwCtrl.cpp
  * \brief The MagAO-X Optec High-Speed Filter Wheel Controller
  *
  * \ingroup hfwCtrl_files
  */


#include "hsfwCtrl.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::hsfwCtrl xapp;

   return xapp.main(argc, argv);
}
