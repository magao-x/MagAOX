#include <map>
#include <random>
#include <iomanip>
#include <iostream>
#include <assert.h>

#include "resurrector.hpp"

int
main(int argc, char** argv)
{
    resurrectorT<> resurr;

    // Parse command-line options
    for (int iargc=1; iargc < argc; ++iargc)
    {
        std::string arg{argv[iargc]};

        size_t iequ = arg.find_first_of('=');
        if (iequ == std::string::npos) { continue; }
        if (iequ == (arg.size()-1)) { continue; }
        if (iequ == 0) { continue; }

        std::string driver_name(arg.substr(0,iequ));
        std::string argv0(arg.substr(iequ+1));
#       ifdef __RESURRECTOR_DEBUG__
        std::cerr
        << '[' << argv0
        << ',' << driver_name
        << "]\n";
#       endif//__RESURRECTOR_DEBUG__
        int newfd = resurr.open_hexbeater(argv0, driver_name, "./fifos", NULL);
        //int pid = resurr.start_hexbeater(newfd);
        resurr.start_hexbeater(newfd);
    }

    do
    {
        struct timeval tv{1,0};
        resurr.srcr_cycle(&tv);
    } while (1);

    return 0;
}
