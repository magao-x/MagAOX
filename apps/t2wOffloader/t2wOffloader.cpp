/** \file t2wOffloader.cpp
  * \brief The MagAO-X Tweeter To Woofer Offloader
  *
  * \ingroup t2wOffloader_files
  */

#include "t2wOffloader.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::t2wOffloader xapp;

   return xapp.main(argc, argv);

}
