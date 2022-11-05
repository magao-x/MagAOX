/** \file qhyCtrl.cpp
  * \brief The QHYCCD camera controller
  *
  * \ingroup qhyCtrl_files
  */

#include "qhyCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::qhyCtrl xapp;

   return xapp.main(argc, argv);

}
