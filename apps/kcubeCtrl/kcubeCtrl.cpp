/** \file kcubeCtrl.cpp
  * \brief The MagAO-X K-Cube Controller main program source file.
  *
  * \ingroup kcubeCtrl_files
  */

#include "kcubeCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::kcubeCtrl xapp;

   return xapp.main(argc, argv);

}
