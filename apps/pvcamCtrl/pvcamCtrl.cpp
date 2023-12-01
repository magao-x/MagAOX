/** \file pvcamCtrl.cpp
  * \brief The MagAO-X pvcam controller main program source file.
  *
  * \ingroup pvcamCtrl_files
  */

#include "pvcamCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::pvcamCtrl xapp;

   return xapp.main(argc, argv);

}
