/** \file observerCtrl.cpp
  * \brief The MagAO-X observer controller main program source file.
  *
  * \ingroup observerCtrl_files
  */

#include "observerCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::observerCtrl xapp;

   return xapp.main(argc, argv);

}
