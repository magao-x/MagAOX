/** \file sysMonitor.cpp
  * \brief The MagAO-X sysMonitor app source file.
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup sysMonitor_files
  *
  * History:
  * - 2018-08-10 created by CJB
  */
#include "sysMonitor.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::sysMonitor mon;

   return mon.main(argc, argv);
}
