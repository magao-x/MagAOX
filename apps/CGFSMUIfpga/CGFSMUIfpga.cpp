/** \file CGFSMUIfpga.cpp
  * \brief The MagAO-X CGFSMUIfpga main program source file.
  * \details INDI-controlled User interface to drive the CGFSMHIfpga
  *          MagAO-X app, intended as proof of concept and not for
  *          typical operational use
  *
  * \ingroup CGFSMUIfpga_files
  */

#include "CGFSMUIfpga.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::CGFSMUIfpga xapp;

   return xapp.main(argc, argv);

}
