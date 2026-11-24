/** \file elliptecCtrl.cpp
  * \brief The MagAO-X Rotation Stage Controller
  *
  * \ingroup elliptecCtrl_files
  */


#include "elliptecCtrl.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::elliptecCtrl rsc;

   return rsc.main(argc, argv);
}

