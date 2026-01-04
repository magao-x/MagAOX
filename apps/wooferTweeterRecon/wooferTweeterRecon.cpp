/** \file wooferTweeterRecon.cpp
  * \brief The MagAO-X Tweeter To Woofer Offloader
  *
  * \ingroup wooferTweeterRecon_files
  */

#include "wooferTweeterRecon.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::wooferTweeterRecon xapp;

   return xapp.main(argc, argv);

}
