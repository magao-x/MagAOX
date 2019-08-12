/** \file zaberCtrl.cpp
  * \brief The MagAO-X Zaber Stage Controller main program source file.
  *
  * \ingroup zaberCtrl_files
  */

#include "zaberCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::zaberCtrl xapp;

   return xapp.main(argc, argv);

}
