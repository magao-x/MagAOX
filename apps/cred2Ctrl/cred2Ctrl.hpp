/** \file cred2Ctrl.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup cred2Ctrl_files
  */

#ifndef cred2Ctrl_hpp
#define cred2Ctrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup cred2Ctrl
  * \brief The XXXXXX application to control the CRED2 camera.
  *
  * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup cred2Ctrl_files
  * \ingroup cred2Ctrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/**
  * \ingroup cred2Ctrl
  */
class cred2Ctrl : public MagAOXApp<true>
{

   //Give the test harness access.
   friend class cred2Ctrl_test;

protected:

   /** \name Configurable Parameters
     *@{
     */

   //here add parameters which will be config-able at runtime

   ///@}




public:
   /// Default c'tor.
   cred2Ctrl();

   /// D'tor, declared and defined for noexcept.
   ~cred2Ctrl() noexcept
   {}

   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   virtual void loadConfig();

   /// Startup function
   /**
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for cred2Ctrl.
   /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int appLogic();

   /// Shutdown the app.
   /**
     *
     */
   virtual int appShutdown();


};

cred2Ctrl::cred2Ctrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{

   return;
}

void cred2Ctrl::setupConfig()
{
}

int cred2Ctrl::loadConfigImpl( mx::app::appConfigurator & _config )
{


   return 0;
}

void cred2Ctrl::loadConfig()
{
   loadConfigImpl(config);
}

int cred2Ctrl::appStartup()
{

   return 0;
}

int cred2Ctrl::appLogic()
{
   return 0;
}

int cred2Ctrl::appShutdown()
{
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //cred2Ctrl_hpp
