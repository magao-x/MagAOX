/** \file loPredCtrlAcc.cpp
  * \brief The MagAO-X low-order predictive controller main program source file.
  *
  * \ingroup loPredCtrlAcc_files
  */

#include "loPredCtrlAcc.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::loPredCtrlAcc xapp;

   return xapp.main(argc, argv);

}
