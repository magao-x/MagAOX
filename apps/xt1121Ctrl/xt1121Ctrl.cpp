/** \file xt1121Ctrl.hpp
  * \brief The MagAO-X Acromag XT 1211 digital I/O controller main program
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup xt1121Ctrl_files
  */


#include "xt1121Ctrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::xt1121Ctrl xtc;

   return xtc.main(argc, argv);

}
