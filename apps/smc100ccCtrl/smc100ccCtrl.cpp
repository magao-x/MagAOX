/** \file smc100ccCtrl.cpp
  * \brief The MagAO-X smc100ccCtrl app source file.
  * \author Chris Bohlman (cbohlman@pm.me)
  *
  * \ingroup smc100ccCtrl_files
  *
  * History:
  * - 2019-01-10 created by CJB
  */
#include "smc100ccCtrl.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::smc100ccCtrl ctrl;

   return ctrl.main(argc, argv);
}
