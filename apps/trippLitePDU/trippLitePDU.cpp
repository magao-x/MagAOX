/** \file trippLitePDU.cpp
  * \brief The MagAO-X Tripp Lite Power Distribution Unit controller main program.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  * 
  * \ingroup trippLitePDU_files
  */


#include "trippLitePDU.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::trippLitePDU pdu;

   return pdu.main(argc, argv);
}
