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
    if (narg != 2) { return 0; }
    if (!*argname || !*argexec) { return 0; }
    if ('#' == *argname || '#' == *argexec) { return 1; }
    name = std::string(argname);
    exec = std::string(argexec);
    return 2;
}

void
Usage(int rtn, const char* msg)
{
    std::cout << msg << "\n"
                 "Usage:\n"
                 "\n"
                 "    [MAGAOX_ROLE=role] resurrector_indi"
                 " [-r role] [--role=role] [-h|--help]\n"
                 "\n" << std::endl;
    exit(rtn);
}

bool
get_verbose_arg(int argc, char** argv)
{
    for (char** av=argv+argc-1; av > argv; --av)
    {
        if (strcmp(*av, "-v")) { continue; }
        return true;
    }
    return false;
}

std::string
get_magaox_proclist_role(int argc, char** argv)
{
    char* role{nullptr};
    char* rptr{nullptr};
    for (char** av=argv+argc-1; av > argv; --av)
    {
        if (!role)
        {
            // Parse [-r the_role]
            if (!strcmp(*av, "-r")) {
                if (!rptr)
                {
                    Usage(1, "ERROR:  -r cannot be the last token");
                }
                role = rptr; 
            }

            // Parse [--role=the_role]
            if (!strncmp(*av, "--role=", 7)) { role = (*av) + 7; }

            rptr = *av;
        }

        if (!strcmp(*av, "-h")) { Usage(0, ""); }
        if (!strcmp(*av, "--help")) { Usage(0, ""); }

    }
    if (!role) { role = getenv("MAGAOX_ROLE"); }
    if (!role) { Usage(1, "\nERROR:  no role specified; try --help"); }

    std::string pl_role{IRMAGAOX_config};  // /opt/MagAOX/config
    pl_role += "/proclist_";
    pl_role += role;
    pl_role += ".txt";

    return pl_role;
}

#endif// __INDI_RESURRECTOR_HPP__
