/** \file bmcCtrl.cpp
  * \brief The MagAO-X xxxxx main program source file.
  *
  * \ingroup bmcCtrl_files
  */

#include "bmcCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::bmcCtrl xapp;

   return xapp.main(argc, argv);

}
