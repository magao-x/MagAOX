/** \file modalGainOpt.cpp
  * \brief The MagAO-X PSD-based gain optimizer main program source file.
  *
  * \ingroup template_files
  */

#include "modalGainOpt.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::modalGainOpt xapp;

   return xapp.main(argc, argv);

}
