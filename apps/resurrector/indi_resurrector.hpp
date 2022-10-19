#ifndef __INDI_RESURRECTOR_HPP__
#define __INDI_RESURRECTOR_HPP__
#include "resurrector.hpp"

#define IRMAGAOX_top "/opt/MagAOX"
#define IRMAGAOX_config IRMAGAOX_top "/config"
#define IRMAGAOX_drivers IRMAGAOX_top "/drivers"
#define IRMAGAOX_bin IRMAGAOX_top "/bin"
#define IRMAGAOX_fifos IRMAGAOX_drivers "/fifos"

int
read_next_process(FILE* f, std::string& name, std::string& exec)
{
    char oneline[1025] = { "" };
    char argname[1025] = { "" };
    char argexec[1025] = { "" };
    char argxtra[1025] = { "" };
    errno = 0;
    char* p = fgets(oneline, 1025, f);
    if (!p) { return EOF; }
    int narg = sscanf(oneline,"%s %s %s",argname,argexec,argxtra);
    // There may be three arguments on whole line, but we want the lines with 2.
    if (narg != 2) { return 0; }
    if (!*argname || !*argexec) { return 0; }
    if ('#' == *argname || '#' == *argexec) { return 1; }
    name = std::string(argname);
    exec = std::string(argexec);
    return 2;
}

#endif// __INDI_RESURRECTOR_HPP__
