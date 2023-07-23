/** \file imgChar.cpp
  * \brief The MagAO-X Image Registrator and Strehl
  * ratio monitor
  *
  * \ingroup imgChar_files
  */

#include "imgChar.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::imgChar xapp;

   return xapp.main(argc, argv);

}
