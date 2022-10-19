/** \file pi335Ctrl.cpp
  * \brief The MagAO-X xxxxx main program source file.
  *
  * \ingroup pi335Ctrl_files
  */

#include "pi335Ctrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::pi335Ctrl xapp;

   return xapp.main(argc, argv);

}
