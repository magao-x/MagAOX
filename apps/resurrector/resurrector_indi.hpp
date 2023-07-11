#ifndef __INDI_RESURRECTOR_HPP__
#define __INDI_RESURRECTOR_HPP__
#include "resurrector.hpp"

//#define IRMAGAOX_top "/opt/MagAOX"
#define IRMAGAOX_top IRMAGAOX_top_func()+
#define IRMAGAOX_config (IRMAGAOX_top "/config")
#define IRMAGAOX_bin (IRMAGAOX_top "/bin")
#define IRMAGAOX_fifos (IRMAGAOX_top "/drivers/fifos")

const std::string&
IRMAGAOX_top_func()
{
    static std::string top{"/opt/MagAOX"};
    static bool once{false};
    if (once) return top;
    char* ge = getenv("MagAOX_PATH");
    if (ge) { top = std::string(ge); }
    once = true;
    return IRMAGAOX_top_func();
}

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
                 " [-r role] [--role=role]"
                 " [-nor|--no-output-redirect]"
                 " [-v|--verbose]"
                 " [-h|--help]\n"
                 "\n" << std::endl;
    exit(rtn);
}

/** \brief Detect presence of command line argument
  *
  * Detect any command line option in argv that matches either
  * function argument ps or pl
  *
  * \returns true if matching option detected, else false
  * \arg \c argc - command line token count
  * \arg \c argv - pointer to array of command line token pointers
  * \arg \c ps - short form of argument, e.g. "-v", or nullptr
  * \arg \c pl - long form of argument, e.g. "--verbose", or nullptr
  */
bool
arg_is_present(int argc, char** argv, const char* ps, const char* pl)
{
    for (char** av=argv+argc-1; av > argv; --av)
    {
        if (ps && !strcmp(*av, ps)) return true;
        if (pl && !strcmp(*av, pl)) return true;
    }
    return false;
}

/** \brief Return value of command line argument
  *
  * Return specified value of command line option in argv that matches
  * either function argument pshrt or plngpfx
  *
  * N.B. for the long form with an equals sign, e.g. --option=value, the
  *      prefix must end with an equals sign e.g. plngpfx => "--option="
  *
  * \returns pointer to value of last matching argument
  * \arg \c argc - command line token count
  * \arg \c argv - pointer to array of command line token pointers
  * \arg pshrt - short form of 2-token option, e.g. -r vm, or nullptr
  * \arg plngpfx - long form of option, e.g. --role=vm, or nullptr
  */
char*
arg_value(int argc, char** argv, const char* pshrt, const char* plngpfx)
{
    char* lastarg{nullptr};
    size_t L{strlen(plngpfx)};
    for (char** av=argv+argc-1; av > argv; --av)
    {
        // Parse [<short> lastarg] over two consecutive args in argv
        if (pshrt && !strcmp(*av, pshrt))
        {
            if (!lastarg)
            {
                Usage(1, (std::string("ERROR:  >>>")
                         +std::string(pshrt)
                         +std::string("<<< cannot be the last token")
                         ).c_str());
            }
            return lastarg;
        }

        // Parse <plngpfx>value
        if (plngpfx && !strncmp(*av, plngpfx, L)) { return (*av) + L; }

        // Save this arg for short form test on next pass through loop
        lastarg = *av;

        if (!strcmp(*av, "-h")) { Usage(0, ""); }
        if (!strcmp(*av, "--help")) { Usage(0, ""); }

    }
    return nullptr;
}

/** \brief Return MagAOX process list role (e.g. vm; RTC; etc.)
  *
  * \returns std::string with role
  * \arg \c argc - command line token count
  * \arg \c argv - pointer to array of command line token pointers
  */
std::string
get_magaox_proclist_role(int argc, char** argv)
{
    char* role{arg_value(argc, argv, "-r", "--role=")};
    if (!role) { role = getenv("MAGAOX_ROLE"); }
    if (!role) { Usage(1, "\nERROR:  no role specified; try --help"); }

    std::string pl_role{IRMAGAOX_config};  // /opt/MagAOX/config
    pl_role += "/proclist_";
    pl_role += role;
    pl_role += ".txt";

    return pl_role;
}

/** \brief Return whether a verbose command-line option is present
  *
  * \returns true if -v or --verbose is present, else false
  * \arg \c argc - command line token count
  * \arg \c argv - pointer to array of command line token pointers
  */
bool
get_verbose_arg(int argc, char** argv)
{
    return arg_is_present(argc, argv, "-v", "--verbose");
}

/** \brief Return whether output redirect is inhibited
  *
  * \returns true if -nor or --no-output-redirect is present, else false
  * \arg \c argc - command line token count
  * \arg \c argv - pointer to array of command line token pointers
  */
bool
get_no_output_redirect_arg(int argc, char** argv)
{
    return arg_is_present(argc, argv, "-nor", "--no-output-redirect");
}


#endif// __INDI_RESURRECTOR_HPP__
