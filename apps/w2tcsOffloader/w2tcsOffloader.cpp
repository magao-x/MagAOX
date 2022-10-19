/** \file w2tcsOffloader.cpp
  * \brief The MagAO-X Woofer To Telescope Control System (TCS) Offloader
  *
  * \ingroup w2tcsOffloader_files
  */

#include "w2tcsOffloader.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::w2tcsOffloader xapp;

   return xapp.main(argc, argv);

}
