/** \file userGainCtrl.cpp
  * \brief The MagAO-X user gain control app
  *
  * \ingroup userGainCtrl_files
  */

#include "userGainCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::userGainCtrl xapp;

   return xapp.main(argc, argv);

}
