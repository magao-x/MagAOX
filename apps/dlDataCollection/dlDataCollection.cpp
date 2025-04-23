/** \file dlDataCollection.cpp
  * \brief The MagAO-X ImageStreamIO integrator main program source file.
  *
  * \ingroup dlDataCollection_files
  */

#include "dlDataCollection.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::dlDataCollection xapp;

   return xapp.main(argc, argv);

}
