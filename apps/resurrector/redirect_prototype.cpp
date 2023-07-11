/** \file MagAOXPathUtil.hpp
  * \brief The basic MagAO-X Application
  * \author Brian T. Carcich (BrianTCarcich@gmail.com)
  *
  * History:
  * - 2023-02-20 created by BTC
  *
  * \ingroup app_files
  */
#include <vector>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/types.h>

// -I../../libMagAOX/common
#include "paths.hpp"
#include "environment.hpp"

/// A class to handle directory names for MagAO-X applications.
/**
  * \ingroup magaoxapp
  */
class MagAOXPathUtil
{
private:

    static std::string magaox_base;  ///< MagAOX base path, typicallly "/ope/MagAOX" from MAGAOX_path macro
                                     ///< or from either from environment variable "MagAOX_PATH|

    static bool singleton;           ///< Flag to ensure magaox_base is assigned but once

    typedef std::vector<std::string> strvec;
    typedef strvec::iterator strvecit;

public:

    /// Constructor:  initialize class-static data (magaox_base), once
    MagAOXPathUtil()
    {
        if (MagAOXPathUtil::singleton) { return; }             // once!
        MagAOXPathUtil::singleton = true;

        // Check MagAOX base directory override environment variable:
        // - if envvar exists,  then use its value;
        // - otherwise use MAGAOX_path macro (typically /opt/MagAOX)
        char* ge = getenv(MAGAOX_env_path);
        MagAOXPathUtil::magaox_base = ge ? ge : MAGAOX_path;
    }

    /**
     * \brief Break down paths into individual sub-directory names,
     *
     * Prepare for equivalent of \c mkdir \c -p
     *
     * Take null-terminated char* vararg strings that are to be
     * concatenated with slashes to represent a directory path, extract
     * individual sub-directory names in the hierarchary.
     *
     * E.g. \c  build("a/b","c",nullptr) will return \c {"a","b","c"}
     *
     * N.B. leading and trailing slashes are ignored
     *
     * \returns vector of individual sub-directory namess
     * \arg \c arg0[in] the first char* string to parse
     * \arg \c args[in] subsequent char* varargs, terminated by nullptr
     */
    strvec const
    build(char* arg0, va_list args)
    {
        strvec rtn{};
        char* p{arg0};
        while (p)
        {
            while (*p=='/') { ++p; }
            if (*p)
            {
                char* pend = p;
                while (*pend && *pend!='/') { ++pend; }
                if (pend != p) { rtn.push_back(std::string(p,pend-p)); }
                p = pend;
                continue;
            }
            p = va_arg(args,char*);
        }
        return rtn;
    }

    /** Build directory paths from individual components
      *
      * on return - errno will be zero on success, non-zero on failure
      *           - returned composite may be incomplete of failure
      *
      * Individual arguments are (char*) pointers
      * - Base MagAOX directory (/opt/MagAOX) prefix is assumed
      * - arguments will be split at slashes (/) by build(...) above
      * - leading, trailing, and duplicate slashes will be removed
      *
      * \return - composite directory path
      * \arg \c mode[in] - mode_t permission bitss for created directories.  N.B. x (execute) will be added to each user/group/other triplet ir r(ead) or w(rite) is 1
      */
    std::string const
    make_dirs(mode_t mode, char* arg0, ...)
    {
        mode_t old_umask{umask(0)};        /// Set 0 umask; save old val

        mode |= (mode&0006) ? 0001 : 000;  /// Set x perm bits as needed
        mode |= (mode&0060) ? 0010 : 000;
        mode |= (mode&0600) ? 0100 : 000;

        /// Init list with MagAOX base path (/opt/MagAOX, if no envvar)
        strvec toks(1,MagAOXPathUtil::magaox_base);

        /// Extract sub-directory names from vararg arguments (...)
        va_list args;
        va_start(args,arg0);
        strvec motoks = build(arg0, args);
        va_end(args);

        /// Append arguments' names to list
        toks.insert(toks.begin()+1,motoks.begin(),motoks.end());

        std::string fullpath{""};  ///< Path to be built and returned

        /// Build path from list, create dirs that do not yet exist
        /// N.B. this loop will exit if errno is non-zero
        errno = 0;
        for (strvecit it=toks.begin(); !errno && it!=toks.end(); ++it)
        {
            /// Append next (or first) sub-directory token
            fullpath += (fullpath.empty() ? "" : "/") + *it;

            struct stat stbuf{0};

            /// Get directory status; true if stat(...) fails
            if (stat(fullpath.c_str(),&stbuf))
            {
                /// If directory does not (yet) exist, ...
                if (errno==ENOENT)
                {
                    /// ... then try to create sub-directory
                    errno = 0;
                    mkdir(fullpath.c_str(),mode);
                    if (errno) { perror("MagAOXPathUtils::make_dirs, mkdir error"); }
                }
                /// Recycle loop on any other stat(...) failure
                continue;
            }

            /// To here, stat(...) did not fail, dir (or file) exists;
            /// ensure it is a directory
            if (S_IFDIR != (stbuf.st_mode & S_IFMT))
            {
                errno = ENOTDIR;
                break;
            }
        }
        if (errno) { perror("MagAOXPathUtils::make_dirs, outside loop"); }

        /// Restore mode (umask), but save errno from stat/mkdir calls
        int save_errno = errno;
        umask(old_umask);
        errno = save_errno;

        /// Return complete path
        return errno ? "" : fullpath;
    } // make_dirs(mode_t mode, char* arg0, ...)
    ////////////////////////////////////////////////////////////////////


    /** Wrapper for make_dirs(...), for /opt/MagAOX/sys/devicename/
      *
      * The created directory will contain file [pid], plus a file with
      * any redirected output
      *
      * \return - complete path
      * \arg \c mode[in] - Permissions mode bits
      * \arg \c device[in] - INDI device name
      */
    std::string  const
    make_sys_device_dirs(mode_t mode, std::string devicename)
    {
        return make_dirs(mode
                        ,(char*)MAGAOX_sysRelPath
                        ,(char*)devicename.c_str()
                        ,nullptr
                        );
    }
    ////////////////////////////////////////////////////////////////////

}; // class MagAOXPathUtil

/// One-time initialization for MagAOXPathUtil static variables;
/// - Assume no environment variable MagAOX_PATH
bool MagAOXPathUtil::singleton = false;
std::string MagAOXPathUtil::magaox_base = MAGAOX_path;
////////////////////////////////////////////////////////////////////////

/** \brief Routine to redirect STDOUT and STDERR to a device under the
  *        MagAOX system directory (/opt/MagAOX/sys/devicename/outputs)
  *
  * \returns - none, but errno will be non-zero if any system calls fail
  * \arg \c devicename[in] - INDI driver device name
  */
void
stdout_stderr_redirect(std::string devicename)
{
    std::string fullpath{""};   // Path to output file
    int fstdxxx{-1};            // File descriptor for open output file
    int old_umask{0};           // Saved umask
    int fopts{O_CREAT|O_TRUNC|O_WRONLY|O_DSYNC};  // open(2) flags

    enum STEPS
    { S_BUILD_SUBDIR = 0  // Build /opt/MagAOX/sys/devicename dir path
    , S_CHANGE_UMASK      // Change umask temporarily
    , S_OPEN_FSTDXXX      // Open file in /opt/MagAOX/sys/devicename
    , S_RESTOR_UMASK      // Restore umask
    , S_REDIR_STDOUT      // Redirect STDOUT to that file
    , S_REDIR_STDERR      // Redirect STDERR to that file
    , S_CLOS_FSTDXXX      // Close open file descriptor
    , S_LAST              // End of loop
    };

    errno = 0;            // Ensure there is no current error

    // Loop over those steps, exit loop early if errno becomes non-zero
    for (int step = S_BUILD_SUBDIR
        ; !errno && step < S_LAST
        ; ++step
        )
    {
        switch(step)
        {
        case S_BUILD_SUBDIR:
            // Build directory path, append output filename
            fullpath = MagAOXPathUtil()
                       .make_sys_device_dirs(0644 ,devicename)
                     + "/outputs";
            break;
        case S_CHANGE_UMASK: old_umask = umask(0); break;
        case S_OPEN_FSTDXXX: fstdxxx = open(fullpath.c_str(),fopts,0644); break;
        case S_RESTOR_UMASK: umask(old_umask); break;
        case S_REDIR_STDOUT: dup2(fstdxxx, STDOUT_FILENO); break;
        case S_REDIR_STDERR: dup2(fstdxxx, STDERR_FILENO); break;
        case S_CLOS_FSTDXXX: close(fstdxxx); break;
        default: break;
        }
    }
} // void stdout_stderr_redirect(std::string devicename)


/** \brief Test harness subroutine for stdout_stderr_redirect above
  *
  * Test harness subroutine to read newline-terminated data from STDIN
  * and write those data to either STDOUT and STDERR
  */
void
TEST_forked_child()
{
  bool use_cerr{false};  // Alternate writing to STDOUT and to STDERR
  while (1)
  {
    std::string s;
    getline(std::cin,s);            // Read line into string
    if (std::cin.eof()) { break; }  // Exit loop on EOF (Control-D?)

    // Write data either to cerr/STDERR or to cout/STDOUT
    if (use_cerr)
    {
        std::cerr << "(std::cerr)" << s;
        write(STDERR_FILENO,"(STDERR_FILENO)",15);
        write(STDERR_FILENO,s.c_str(), s.length());
        fprintf(stderr, "(stderr)%s",s.c_str());
    }
    else
    {
        std::cout << "std::cout[" << s << "]" << std::endl;
        write(STDOUT_FILENO,"STDOUT_FILENO[",14);
        write(STDOUT_FILENO,s.c_str(),s.length());
        write(STDOUT_FILENO,"]\n",2);
        fprintf(stdout, "stdout[%s]\n",s.c_str());
    }
    use_cerr ^= true;  // Alternate writing to STDOUT and to STDERR
  }
}

/** \brief Test harness main routine for stdout_stderr_redirect above
  *
  * Test harness to exercise stdout_stderr_redirect.
  *
  * With one argument (argc=2; argv[1]=devicnname), parent process will
  * fork a child and then wait for that child to exit.  Child process
  * redirects it STDOUT and STDERR output to a file with full path of
  * /opt/MagAOX/sys/<devicename>/outputs, and then run this app with
  * no arguments.
  *
  * With no argument, (i.e. forked child), this will run the subroutine
  * TEST_forked_child() above to send test data to STDOUT and to STDERR,
  * which test data should be redirected to /opt/MagAOX/sys/.../outputs
  */
int
TEST_MAIN(int argc, char** argv)
{
  pid_t pid{0};
  pid_t pidwait {0};
  int wstatus{0};
  fprintf(stderr,"%d,%lu={argc,getpid()}\n"
         ,argc,(unsigned long)getpid());

  // Run test data to redirected outputs, and exit
  if (argc==1) { TEST_forked_child(); exit(errno ? 1 : 0); }

  if (argc==2)
  {
    errno = 0;
    if (!(pid=fork()))
    {
      fprintf(stderr,"%d,%lu={argc,getpid()}\n"
             ,argc,(unsigned long)getpid());
      stdout_stderr_redirect(argv[1]);
      execlp(argv[0], argv[0], NULL);
      exit(errno ? 2 : 0);
    }
  }

  while ((pidwait = waitpid(pid,&wstatus,0)) && errno==EINTR)
  {
    errno = 0;
  }

  fprintf(stderr,"waitpid(%ld,&wstatus,0) returned %ld"
                 "; %d=wstatus(%s); errno=%d\n"
         ,(unsigned long)pid,(unsigned long)pidwait
         ,wstatus,strerror(wstatus),errno
         );
  return 0;
}
