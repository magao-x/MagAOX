/** \file filterWheelCtrl.cpp
  * \brief The MagAO-X Filter Wheel Controller
  *
  * \ingroup filterWheelCtrl_files
  */


#include "filterWheelCtrl.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::filterWheelCtrl fwc;

   return fwc.main(argc, argv);
}
