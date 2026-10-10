/** \file streamWriter.cpp
 * \brief Main entrypoint for the MagAO-X image stream writer.
 * \ingroup streamWriter_files
 */
#include "streamWriter.hpp"

/// Run the stream writer application.
int main(int argc /**< [in] argument count */, char ** argv /**< [in] command-line arguments */)
{
   MagAOX::app::streamWriter sw;

   return sw.main(argc, argv);
}
