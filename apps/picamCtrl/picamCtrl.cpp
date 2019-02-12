/** \file picamCtrl.cpp
  * \brief The MagAO-X Princeton Instruments EMCCD camera controller main program.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  * 
  * \ingroup picamCtrl_files
  */


#include "picamCtrl.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::picamCtrl pcam;

   return pcam.main(argc, argv);
}
