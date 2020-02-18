/** \file koolanceCtrl.cpp
  * \brief The MagAO-X Koolance Controller main program source file.
  *
  * \ingroup koolanceCtrl_files
  */

#include "koolanceCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::koolanceCtrl xapp;

   return xapp.main(argc, argv);

}
