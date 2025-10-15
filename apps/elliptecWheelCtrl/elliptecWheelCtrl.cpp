/** \file elliptecWheelCtrl.cpp
  * \brief The MagAO-X Filter Wheel Controller
  *
  * \ingroup elliptecWheelCtrl_files
  */


#include "elliptecWheelCtrl.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::elliptecWheelCtrl ewc;

   return ewc.main(argc, argv);
}
