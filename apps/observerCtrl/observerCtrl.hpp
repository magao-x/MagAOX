/** \file observerCtrl.hpp
  * \brief The MagAO-X Observer Controller header file
  *
  * \ingroup observerCtrl_files
  */

#ifndef observerCtrl_hpp
#define observerCtrl_hpp

#include <map>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup observerCtrl
  * \brief The MagAO-X Observer Controller application
  *
  * <a href="../handbook/operating/software/apps/observerCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup observerCtrl_files
  * \ingroup observerCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X Observer Controller
/** 
  * \ingroup observerCtrl
  */
class observerCtrl : public MagAOXApp<true>, public dev::telemeter<observerCtrl>
{

   //Give the test harness access.
   friend class observerCtrl_test;
   
   friend class dev::telemeter<observerCtrl>;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::vector<std::string> m_streamWriters; ///< The stream writers to stop and start
   
   ///@}

   struct observer
   {
      std::string m_fullName;
      std::string m_pfoa;
      std::string m_email;
      std::string m_sanitizedEmail;
      std::string m_institution;
   };

   typedef std::map<std::string, observer> observerMapT;
   
   observerMapT m_observers;
   
   observer m_currentObserver;
   
   std::string m_obsName;
   bool m_observing {false};
   
   

public:
   /// Default c'tor.
   observerCtrl();

   /// D'tor, declared and defined for noexcept.
   ~observerCtrl() noexcept
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

   /// Implementation of the FSM for observerCtrl.
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

   void startObserving();

   void stopObserving();

   ///\name INDI
   /** @{
     */ 
protected:
   pcf::IndiProperty m_indiP_observers;
   pcf::IndiProperty m_indiP_observer;
   
   pcf::IndiProperty m_indiP_obsName;
   pcf::IndiProperty m_indiP_observing;
   pcf::IndiProperty m_indiP_obslog;
   
   pcf::IndiProperty m_indiP_sws;
   

public:
   INDI_NEWCALLBACK_DECL(observerCtrl, m_indiP_observers);
   
   INDI_NEWCALLBACK_DECL(observerCtrl, m_indiP_obsName);
   INDI_NEWCALLBACK_DECL(observerCtrl, m_indiP_observing);
   
   INDI_NEWCALLBACK_DECL(observerCtrl, m_indiP_sws);

   ///@}
   
    /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_observer * );
      
   int recordObserver(bool force = false);
   
   int recordObserverNow();
   
   ///@}
};

observerCtrl::observerCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void observerCtrl::setupConfig()
{
   config.add("stream.writers", "", "stream.writers", argType::Required, "stream", "writers", false, "string", "The device names of the stream writers to control.");

   dev::telemeter<observerCtrl>::setupConfig(config);
}

int observerCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_streamWriters, "stream.writers");

   std::vector<std::string> sections;

   _config.unusedSections(sections);

   if( sections.size() == 0 )
   {
      log<text_log>("no observers found in config", logPrio::LOG_CRITICAL);
      return -1;
   }
   
   for(size_t i=0; i< sections.size(); ++i)
   {
      bool pfoaSet = _config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "pfoa" ));
      if( !pfoaSet ) continue;
      
      std::string email = sections[i];
      
      std::string pfoa;
      _config.configUnused(pfoa, mx::app::iniFile::makeKey(sections[i], "pfoa" ));
      
      std::string fullName;
      _config.configUnused(fullName, mx::app::iniFile::makeKey(sections[i], "full_name" ));
      
      std::string institution;
      _config.configUnused(institution, mx::app::iniFile::makeKey(sections[i], "institution" ));
      
      std::string sanitizedEmail = "";
      for(size_t n = 0; n < email.size(); ++n)
      {
         if(email[n] == '@') {
            sanitizedEmail = sanitizedEmail + "-at-";
         } else if(email[n] == '.') {
            sanitizedEmail = sanitizedEmail + "-dot-";
         } else {
            sanitizedEmail.push_back(email[n]);
         }
      }
      m_observers[email] = observer({fullName, pfoa, email, sanitizedEmail, institution});
   }
   
   return 0;
}

void observerCtrl::loadConfig()
{
   if(loadConfigImpl(config) < 0)
   {
      m_shutdown = 1;
      return;
   }
   
   if(m_observers.size() < 1)
   {
      log<text_log>("no observers found in config", logPrio::LOG_CRITICAL);
      m_shutdown = 1;
      return;
   }
   
   dev::telemeter<observerCtrl>::loadConfig(config);
}

int observerCtrl::appStartup()
{
   std::vector<std::string> sanitizedEmails;
   std::vector<std::string> emails;
   for(auto it = m_observers.begin(); it!=m_observers.end(); ++it)
   {
      sanitizedEmails.push_back(it->second.m_sanitizedEmail);
      emails.push_back(it->second.m_email);
   }
      
   if(createStandardIndiSelectionSw( m_indiP_observers, "observers", sanitizedEmails, emails) < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
    
   if(registerIndiPropertyNew( m_indiP_observers, INDI_NEWCALLBACK(m_indiP_observers)) < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
   
   createStandardIndiText( m_indiP_obsName, "obs_name", "Observation Name", "Observer"); 
   if(registerIndiPropertyNew( m_indiP_obsName, INDI_NEWCALLBACK(m_indiP_obsName)) < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }   
   
   createStandardIndiToggleSw( m_indiP_observing, "obs_on", "Observation On", "Observer");
   if(registerIndiPropertyNew( m_indiP_observing, INDI_NEWCALLBACK(m_indiP_observing)) < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
   
   REG_INDI_NEWPROP_NOCB(m_indiP_observer, "current_observer", pcf::IndiProperty::Text);
   m_indiP_observer.add(pcf::IndiElement("full_name"));
   m_indiP_observer.add(pcf::IndiElement("email"));
   m_indiP_observer.add(pcf::IndiElement("pfoa"));
   m_indiP_observer.add(pcf::IndiElement("institution"));
   
   m_indiP_sws = pcf::IndiProperty(pcf::IndiProperty::Switch);
   m_indiP_sws.setDevice(configName());
   m_indiP_sws.setName("writers");
   m_indiP_sws.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_sws.setState(pcf::IndiProperty::Idle);
   m_indiP_sws.setRule(pcf::IndiProperty::AnyOfMany);

   for(size_t n =0; n < m_streamWriters.size(); ++n)
   {
      m_indiP_sws.add(pcf::IndiElement(m_streamWriters[n], pcf::IndiElement::Off));
   }

   if(registerIndiPropertyNew( m_indiP_sws, INDI_NEWCALLBACK(m_indiP_sws)) < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }

   if(dev::telemeter<observerCtrl>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   state(stateCodes::READY);
   return 0;
}

int observerCtrl::appLogic()
{
   
   std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);
   
   if(lock.owns_lock())
   {
      updateIfChanged<std::string>(m_indiP_observer, {"full_name","email","pfoa","institution"},{m_currentObserver.m_fullName, m_currentObserver.m_email, m_currentObserver.m_pfoa, m_currentObserver.m_institution});
      
      
      for(auto it = m_observers.begin();it!=m_observers.end();++it)
      {
         if(it->first == m_currentObserver.m_email) updateSwitchIfChanged(m_indiP_observers, it->second.m_sanitizedEmail, pcf::IndiElement::On, INDI_IDLE);
         else updateSwitchIfChanged(m_indiP_observers, it->second.m_sanitizedEmail, pcf::IndiElement::Off, INDI_IDLE);
      }
         
      updateIfChanged(m_indiP_obsName, "current", m_obsName);
      updateIfChanged(m_indiP_obsName, "target", m_obsName);
   }
      
   if(telemeter<observerCtrl>::appLogic() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return 0;
   }
   
   return 0;
      
}

int observerCtrl::appShutdown()
{
   dev::telemeter<observerCtrl>::appShutdown();
   
   return 0;
}

void observerCtrl::startObserving()
{
   for(size_t n=0; n < m_streamWriters.size(); ++n)
   {
      if(m_indiP_sws[m_streamWriters[n]].getSwitchState() == pcf::IndiElement::On)
      {
         pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
         ip.setDevice(m_streamWriters[n] + "-sw");
         ip.setName("writing");
         ip.add(pcf::IndiElement("toggle"));
         ip["toggle"].setSwitchState(pcf::IndiElement::On);

         sendNewProperty(ip); 
      }
   }

   mx::sys::sleep(1);

   m_observing = true;
   recordObserver();
}

void observerCtrl::stopObserving()
{
   m_observing = false;
   recordObserver();

   for(size_t n=0; n < m_streamWriters.size(); ++n)
   {
      if(m_indiP_sws[m_streamWriters[n]].getSwitchState() == pcf::IndiElement::On)
      {
         pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
         ip.setDevice(m_streamWriters[n] + "-sw");
         ip.setName("writing");
         ip.add(pcf::IndiElement("toggle"));
         ip["toggle"].setSwitchState(pcf::IndiElement::Off);

         sendNewProperty(ip); 
      }
   }


}

INDI_NEWCALLBACK_DEFN(observerCtrl, m_indiP_observers)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() != m_indiP_observers.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   //look for selected mode switch which matches a known mode.  Make sure only one is selected.
   std::string newEmail = "";
   for(auto it=m_observers.begin(); it != m_observers.end(); ++it) 
   {
      if(!ipRecv.find(it->second.m_sanitizedEmail)) continue;
      
      if(ipRecv[it->second.m_sanitizedEmail].getSwitchState() == pcf::IndiElement::On)
      {
         if(newEmail != "")
         {
            log<text_log>("More than one observer selected", logPrio::LOG_ERROR);
            return -1;
         }
         
         newEmail = it->first;
      }
   }
   
   if(newEmail == "")
   {
      std::cerr << "nothing\n";
      return 0; 
   }
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   m_currentObserver = m_observers[newEmail];
   
   for(auto it = m_observers.begin();it!=m_observers.end();++it)
   {
      if(it->first == m_currentObserver.m_sanitizedEmail) updateSwitchIfChanged(m_indiP_observers, it->second.m_sanitizedEmail, pcf::IndiElement::On, INDI_IDLE);
      else updateSwitchIfChanged(m_indiP_observers, it->second.m_sanitizedEmail, pcf::IndiElement::Off, INDI_IDLE);
   }
   
   log<logger::observer>({m_currentObserver.m_fullName,m_currentObserver.m_pfoa, m_currentObserver.m_email, m_currentObserver.m_institution});
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(observerCtrl, m_indiP_obsName)(const pcf::IndiProperty &ipRecv)
{
   std::string target;

   std::unique_lock<std::mutex> lock(m_indiMutex);

   if( indiTargetUpdate( m_indiP_obsName, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_obsName = target;
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(observerCtrl, m_indiP_observing)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() != m_indiP_observing.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   if(!ipRecv.find("toggle")) return 0;
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   recordObserver(true);
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      startObserving();
      updateSwitchIfChanged(m_indiP_observing, "toggle", pcf::IndiElement::On, INDI_OK);
   }   
   else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      stopObserving();
      updateSwitchIfChanged(m_indiP_observing, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(observerCtrl, m_indiP_sws)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() != m_indiP_sws.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   if(m_observing == true)
   {
      log<text_log>({"Can't change stream writers while observing"}, logPrio::LOG_WARNING);
      return 0;
   }

   for(size_t n =0; n < m_streamWriters.size(); ++n) 
   {
      if(!ipRecv.find(m_streamWriters[n])) continue;
      
     updateSwitchIfChanged(m_indiP_sws, m_streamWriters[n], ipRecv[m_streamWriters[n]].getSwitchState());
   }
      
   return 0;
}

inline
int observerCtrl::checkRecordTimes()
{
   return telemeter<observerCtrl>::checkRecordTimes(telem_observer());
}
   
inline
int observerCtrl::recordTelem( const telem_observer * )
{
   return recordObserver(true);
}

inline
int observerCtrl::recordObserver( bool force )
{
   static std::string last_email;
   static std::string last_obsName;
   static bool last_observing;
   
   if( last_email != m_currentObserver.m_email || last_obsName != m_obsName || last_observing != m_observing || force)
   {
      telem<telem_observer>({m_currentObserver.m_email, m_obsName, m_observing});
      
      last_email = m_currentObserver.m_email;
      last_obsName = m_obsName;
      last_observing = m_observing;
   }
   
   return 0;
}

inline
int observerCtrl::recordObserverNow()
{
   telem<telem_observer>({m_currentObserver.m_email, m_obsName, m_observing});
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //observerCtrl_hpp
