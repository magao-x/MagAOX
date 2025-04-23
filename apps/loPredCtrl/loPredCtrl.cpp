/** \file loPredCtrl.cpp
  * \brief The MagAO-X low-order predictive controller main program source file.
  *
  * \ingroup loPredCtrl_files
  */

#include "loPredCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::loPredCtrl xapp;

   return xapp.main(argc, argv);

}
  