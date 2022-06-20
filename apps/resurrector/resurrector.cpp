#include <map>
#include <random>
#include <iomanip>
#include <iostream>
#include <assert.h>

#include "resurrector.hpp"

#ifndef OLDTEST
#define OLDTEST 0
#endif//ifndef OLDTEST

int
main(int argc, char** argv)
{
    resurrectorT<> resurr;

    
    for (int iargc=1; iargc < argc; ++iargc)
    {
        std::string arg(argv[iargc]);

        int iequ = arg.find_first_of('=');
        if (iequ == std::string::npos) { continue; }
        if (iequ == (arg.size()-1)) { continue; }
        if (iequ == 0) { continue; }

        std::string driver_name(arg.substr(0,iequ));
        std::string argv0(arg.substr(iequ+1));
        std::cerr
        << '[' << argv0
        << ',' << driver_name
        << "]\n";
    }

    return 0;
}

#   if OLDTEST
    int i = 0;
    int fails = 0;
    while (1)
    {
        std::ostringstream oss(std::ios_base::ate);
        oss.str("");
        oss << std::setw(4) << std::setfill('0') << i++;
        int newfd = resurr.add_hexbeat("./something", oss.str(), "./fifos", NULL);
        std::cout << i-1 << ":  [" << newfd << "," << (newfd < 0 ? std::string("-1") : resurr.fifo_name_hbm(newfd)) << "]\n";
        if (newfd < 0) { if (++fails > 2) { break; } }
        else           { fails = 0; }
    }
#   endif//OLDTEST

#   if OLDTEST
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> gaussian{0,5};

    int threesigma = 0;
    std::map<int, int> hist;
    for(int n=0; n<1e6; ++n) {
    double gg = gaussian(gen);
        if (std::abs(gg) < 15.0) { ++threesigma; }
        ++hist[std::round(gg)];
    }
    for(auto p : hist) {
        std::cout << std::setw(3)
                  << p.first << ' ' << std::string((80*p.second)/hist[0], '*')
                  << ' ' << p.second
                  << '\n';
    }
    std::cout << (threesigma/1e4) << "% = threesigma\n";
#   endif//OLDTEST
