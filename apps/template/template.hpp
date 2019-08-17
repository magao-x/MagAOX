/** \file template.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup template_files
  */

#ifndef template_hpp
#define template_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup template
  * \brief The XXXXXX application to do YYYYYYY
  *
  * <a href="../handbook/apps/XXXXXX.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup template_files
  * \ingroup template
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/** 
  * \ingroup template
  */
class template : public MagAOXApp<true>
{

   //Give the test harness access.
   friend class template_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   
   ///@}




public:
   /// Default c'tor.
   template();

   /// D'tor, declared and defined for noexcept.
   ~template() noexcept
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

   /// Implementation of the FSM for template.
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

template::template() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void template::setupConfig()
{
}

int template::loadConfigImpl( mx::app::appConfigurator & _config )
{

   
   return 0;
}

void template::loadConfig()
{
   loadConfigImpl(config);
}

int template::appStartup()
{
   
   return 0;
}

int template::appLogic()
{
   return 0;
}

int template::appShutdown()
{
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //template_hpp
