/** \file xrif2fits.cpp
 * \brief The xrif2fits main program.
 *
 * \ingroup xrif2fits_files
 */

#include "xrif2fits.hpp"

int main( int argc, char **argv )
{
    try
    {
        xrif2fits *xs;
        try
        {
            xs = new xrif2fits;
        }
        catch( const std::exception &e )
        {
            std::throw_with_nested( mx::exception( mx::error_t::std_exception, "error constructing xrif2fits" ) );
        }

        try
        {
            int rv = xs->main( argc, argv );
            if( rv < 0 )
            {
                rv = 2;
            }

            if( rv >= 2 )
            {
                std::cerr << "xrif2fits fatal error. run with -h to seek help.\n";
            }

            return rv;
        }
        catch( ... )
        {
            std::throw_with_nested( mx::exception( "error from mx::app::main" ) );
        }

        try
        {
            delete xs;
        }
        catch( ... )
        {
            std::throw_with_nested( mx::exception( "error during destruction" ) );
        }
    }
    catch( const std::exception &e )
    {
        std::vector<std::string> whats;
        mx::unwind_exceptions( whats, e );
        mx::print_exceptions( whats );

        return -1;
    }
}
