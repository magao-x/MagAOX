/** \file acronameUsbHub.hpp
  * \brief The MagAO-X Acromname USB Hub controller main program
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup acronameUsbHub_files
  */


#include "acronameUsbHub.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::acronameUsbHub xapp;

   return xapp.main(argc, argv);

}
