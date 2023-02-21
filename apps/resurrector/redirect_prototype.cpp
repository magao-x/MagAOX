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

#include "../../libMagAOX/common/paths.hpp"
#include "../../libMagAOX/common/environment.hpp"

#if 0//Unused code:  this redirects STDOUT/ERR for C++-based code only
void
cout_cerr_redirect(int argc, char** argv)
{
  enum STEPS 
  { STEPS_INIT = 0
  , STEPS_CHECK_ARGC
  , STEPS_INIT_UMASK
  , STEPS_OPEN_FCOUT
  , STEPS_OPEN_FCERR
  , STEPS_REST_UMASK
  , STEPS_REDIR_COUT
  , STEPS_REDIR_CERR
  , STEPS_LAST = STEPS_REDIR_CERR
  };
  int step{STEPS_INIT};
  int old_umask{0};
  int tmp_umask{0};
  static std::ofstream fcout;
  static std::ofstream fcerr;
  while (!errno && step < STEPS_LAST)
  {
    ++step;
    switch(step)
    {
    case STEPS_CHECK_ARGC: errno = (argc == 3) ? 0 : EINVAL; break;
    case STEPS_INIT_UMASK: old_umask = umask(0);
    case STEPS_OPEN_FCOUT: fcout.open(argv[1],std::ios_base::trunc); break;
    case STEPS_OPEN_FCERR: fcerr.open(argv[2],std::ios_base::trunc); break;
    case STEPS_REST_UMASK: tmp_umask = umask(old_umask); break;
    case STEPS_REDIR_COUT: std::cout.rdbuf(fcout.rdbuf()); break;
    case STEPS_REDIR_CERR: std::cerr.rdbuf(fcerr.rdbuf()); break;
    default: break;
    }
  }
  if (getenv("REDIRECT_DEBUG"))
  {
    std::cerr
    << "{" << step
    << "," << old_umask
    << "," << tmp_umask
    << "}={step,old_umask,tmp_umask}"
    << std::endl;
  }
}
#endif//0//Unused

/// A class to handle directory names for MagAO-X applications.
/**
  * \todo implement libMagAOX error handling? (a stack?)
  * \todo make m_powerMgtEnabled a template parameter, and static_assert checki if _useINDI== false and power management is true
  *
  * \ingroup magaoxapp
  */
class MagAOXPathUtil
{
private:

    static std::string magaox_base;           ///< This will be either from envvar MagAOX_PATH, or  "/opt/MagAOX" from macro MAGAOX_path

    static bool singleton;                    ///< Only check envvar MagAOX_PATH once

    typedef std::vector<std::string> strvec;
    typedef strvec::iterator strvecit;

public:

    /// Constructor:  initialize class-static data (magaox_base)
    MagAOXPathUtil()
    {
        if (MagAOXPathUtil::singleton) { return; }
        char* ge = getenv(MAGAOX_env_path);
        MagAOXPathUtil::magaox_base = ge ? ge : MAGAOX_path;
        MagAOXPathUtil::singleton = true;
    }

    /// Break down paths into individual directory names
    /** \returns vector of individual sub-directory namess
     **/
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

    /// Build directory paths from individual components
    /** \return - composite directory path
     ** on return - errno will be zero on success, non-zero on failure
     **           - returned composite may be incomplete of failure
     **
     ** Individual arguments are (char*) pointers
     ** - Base MagAOX directory (/opt/MagAOX) prefix is assumed
     ** - arguments will be split at slashes (/) by build(...) above
     ** - leading, trailing, and duplicate slashes will be removed
     **/ 
    std::string const
    make_dirs(mode_t mode, char* arg0, ...)
    {
        mode_t old_umask{umask(0)};        /// Save mode (umask)

        mode |= (mode&0006) ? 0001 : 000;  /// Set x perm bits as needed
        mode |= (mode&0060) ? 0010 : 000;
        mode |= (mode&0600) ? 0100 : 000;

        /// Init list with MagAOX base path (/opt/MagAOX, if no envvar)
        strvec toks(1,MagAOXPathUtil::magaox_base);

        /// Extract single subdirectory names from arguments
        va_list args;
        va_start(args,arg0);
        strvec motoks = build(arg0, args);
        va_end(args);

        /// Append arguments' names to list
        toks.insert(toks.begin()+1,motoks.begin(),motoks.end());

        std::string fullpath{""};  ///< Path that is being built

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
                    /// ... then Try to create subdirectory
                    errno = 0;
                    mkdir(fullpath.c_str(),mode);
                    if (errno) { perror("MagAOXPathUtils::make_dirs, mkdir error"); }
                }
                /// Recycle loop on any stat(...) failure
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


    /// Wrapper for make_dirs(...), for /opt/MagAOX/sys/devicename/
    /** \return - complete path
     ** The created directory will contain file [pid], plus any
     ** redirected output
     **/
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

/// One-time initialization; assume no environment variable MagAOX_PATH
bool MagAOXPathUtil::singleton = false;
std::string MagAOXPathUtil::magaox_base = MAGAOX_path;

/// One-time initialization; assume no environment variable MagAOX_PATH
void
make_magaox_indi_device_dir(mode_t mode, std::string devicename)
{
    std::string fullpath{
        MagAOXPathUtil().make_sys_device_dirs(mode ,devicename)
                        };
    return;
}

void
stdout_stderr_redirect(std::string devicename)
{
    std::string fullpath{""};
    enum STEPS 
    { S_BUILD_SUBDIR = 0, S_CHANGE_UMASK, S_OPEN_FSTDXXX, S_RESTOR_UMASK
    , S_REDIR_STDOUT, S_REDIR_STDERR, S_CLOS_FSTDXXX
    , S_LAST
    };
    int fstdxxx{-1};
    int old_umask{0};
    int fopts{O_CREAT|O_TRUNC|O_WRONLY|O_DSYNC};
    for (int step = S_BUILD_SUBDIR
        ; !errno && step < S_LAST
        ; ++step
        )
    {
        switch(step)
        {
        case S_BUILD_SUBDIR:
            fullpath = MagAOXPathUtil()
                       .make_sys_device_dirs(0644 ,devicename)
                     + "/outputs";
            break;
        case S_CHANGE_UMASK: old_umask = umask(0);
        case S_OPEN_FSTDXXX: fstdxxx = open(fullpath.c_str(),fopts,0644); break;
        case S_RESTOR_UMASK: umask(old_umask); break;
        case S_REDIR_STDOUT: dup2(fstdxxx, STDOUT_FILENO); break;
        case S_REDIR_STDERR: dup2(fstdxxx, STDERR_FILENO); break;
        case S_CLOS_FSTDXXX: close(fstdxxx); break;
        default: break;
        }
    }
}


void
forked_child()
{
  bool use_cout{false};
  do
  {
    std::string s;
    std::cin >> s;
    if (std::cin.eof()) { break; }
    if (use_cout) { std::cerr << "(std::cerr)" << s; write(STDERR_FILENO,"STDERR_FILENO",13); write(STDERR_FILENO,s.c_str(), s.length());  }
    else          { std::cout << "std::cout[" << s << "]" << std::endl; write(STDOUT_FILENO,"STDOUT_FILENO[",14); write(STDOUT_FILENO,s.c_str(),s.length()); write(STDOUT_FILENO,"]\n",2); }
    use_cout ^= true;
  } while (1);
}

int
main(int argc, char** argv)
{
  pid_t pid{0};
  pid_t pidwait {0};
  int wstatus{0};
  fprintf(stderr,"%d,%llx,%lu={argc,argv,getpid()}\n"
         ,argc,(long long unsigned) argv,(unsigned long)getpid());

  if (argc==1) { forked_child(); exit(errno ? 1 : 0); }

  if (argc==2)
  {
    errno = 0;
    if (!(pid=fork()))
    {
      fprintf(stderr,"%d,%llx,%lu={argc,argv,getpid()}\n"
         ,argc,(long long unsigned) argv,(unsigned long)getpid());
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
