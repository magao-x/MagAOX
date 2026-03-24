/**
 * \file cred2Ctrl.cpp
 * \brief MagAO-X C-RED2 camera application source file.
 *
 * \ingroup cred2Ctrl_files
 */

#include "cred2Ctrl.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::cred2Ctrl cred2;

   return cred2.main(argc, argv);
}