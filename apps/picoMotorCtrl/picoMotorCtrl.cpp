/** \file picoMotorCtrl.cpp
  * \brief The MagAO-X Pico Motor Controller main program source file.
  *
  * \ingroup picoMotorCtrl_files
  */

#include "picoMotorCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::picoMotorCtrl xapp;

   return xapp.main(argc, argv);

}
