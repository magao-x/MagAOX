
#ifndef magAOXMaths_hpp
#define magAOXMaths_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <limits>

namespace MagAOX
{
namespace app
{

/*
void externalLog ( const std::string & name,
                         const int & code,
                         const std::string & valueStr,
                         const std::string & source
                       )
{
   std::cerr << name << " " << code << " " << valueStr << " " << source << "\n";
}*/


/** MagAO-X application to do math on some numbers
  *
  */
class magAOXMaths : public MagAOXApp<>
{

protected:
   double m_val {0};

   // declare our properties
   pcf::IndiProperty m_indiP_myVal;
   pcf::IndiProperty m_indiP_myVal_maths;

   pcf::IndiProperty m_indiP_otherVal;

   pcf::IndiProperty m_indiP_setOtherVal;
   
   std::string m_myVal {"x"};
   std::string m_otherDevName;
   std::string m_otherValName;

   int updateVals();

public:

   /// Default c'tor.
   magAOXMaths();

   ~magAOXMaths() noexcept
   {
   }


   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// Load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Checks if the device was found during loadConfig.
   virtual int appStartup();

   /// Implementation of the FSM for the maths.
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();


   INDI_NEWCALLBACK_DECL(magAOXMaths, m_indiP_myVal);

   INDI_SETCALLBACK_DECL(magAOXMaths, m_indiP_otherVal);
   
   INDI_NEWCALLBACK_DECL(magAOXMaths,  m_indiP_setOtherVal);

};

magAOXMaths::magAOXMaths() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void magAOXMaths::setupConfig()
{
   config.add("myVal", "", "myVal", argType::Required, "", "myVal", false, "string", "The name of this app's value.");
   config.add("otherDevName", "", "otherDevName", argType::Required, "", "otherDevName", false, "string", "The name of the other app name.");
   config.add("otherValName", "", "otherValName", argType::Required, "", "otherValName", false, "string", "The name of the other val name.");
}

void magAOXMaths::loadConfig()
{
   config(m_myVal, "myVal");
   config(m_otherDevName, "otherDevName");
   config(m_otherValName, "otherValName");

}

int magAOXMaths::appStartup()
{
   // set up the x input property
   REG_INDI_NEWPROP(m_indiP_myVal, m_myVal, pcf::IndiProperty::Number);
   indi::addNumberElement<double>( m_indiP_myVal, "value",  std::numeric_limits<double>::min(),  std::numeric_limits<double>::max(), 1.0,  "%f", "");
   m_indiP_myVal["value"].set<double>(0.0);


   // set up the result maths property
   REG_INDI_NEWPROP_NOCB(m_indiP_myVal_maths, "maths", pcf::IndiProperty::Number);
   indi::addNumberElement<double>(m_indiP_myVal_maths,"value", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");
   indi::addNumberElement<double>(m_indiP_myVal_maths, "sqr", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");
   indi::addNumberElement<double>(m_indiP_myVal_maths, "sqrt", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");
   indi::addNumberElement<double>(m_indiP_myVal_maths, "abs", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");
   indi::addNumberElement<double>(m_indiP_myVal_maths, "prod", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");

   REG_INDI_SETPROP(m_indiP_otherVal,  m_otherDevName, m_otherValName);
   m_indiP_otherVal.add (pcf::IndiElement("value"));
   m_indiP_otherVal["value"].set<double>(0.0);

   createStandardIndiNumber<double>(  m_indiP_setOtherVal, "other_val", -1e50, 1e50, 0, "%0.f");
    m_indiP_setOtherVal["current"].set<double>(0.0);
    m_indiP_setOtherVal["target"].set<double>(0.0);
   registerIndiPropertyNew( m_indiP_setOtherVal, INDI_NEWCALLBACK( m_indiP_setOtherVal));
   
   updateVals();
   state(stateCodes::READY);
   return 0;
}

int magAOXMaths::appLogic()
{
   return 0;

}

int magAOXMaths::appShutdown()
{

   return 0;
}

int magAOXMaths::updateVals()
{
   // extract value
   double v = m_indiP_myVal["value"].get<double>();

   if(v == -1) log<text_log>( "value set to -1!", logPrio::LOG_WARNING);
   if(v == -2) log<text_log>( "value set to -2!", logPrio::LOG_ERROR);
   if(v == -3) log<text_log>( "value set to -3!", logPrio::LOG_CRITICAL);
   if(v == -4) log<text_log>( "value set to -4!", logPrio::LOG_ALERT);
   if(v == -5) log<text_log>( "value set to -5!", logPrio::LOG_EMERGENCY);
   
   // fill maths
   m_indiP_myVal_maths["value"] = v;
   m_indiP_myVal_maths["sqr"] = v*v;
   m_indiP_myVal_maths["sqrt"] = sqrt(v);
   m_indiP_myVal_maths["abs"] = fabs(v);

   m_indiP_myVal_maths["prod"] = v*m_indiP_otherVal["value"].get<double>();
   updateIfChanged( m_indiP_setOtherVal, "current", m_indiP_otherVal["value"].get<double>());

   log<text_log>("set new value: " + std::to_string(v), logPrio::LOG_NOTICE);
   // publish maths
   m_indiP_myVal_maths.setState (pcf::IndiProperty::Ok);
   if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_myVal_maths);

   return 0;
}

INDI_NEWCALLBACK_DEFN(magAOXMaths, m_indiP_myVal)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() == m_indiP_myVal.getName())
   {
      // received a new value for property val
      m_indiP_myVal["value"] = ipRecv["value"].get<double>();
      m_indiP_myVal.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (m_indiP_myVal);

      updateVals();

      return 0;
   }
   return -1;
}

INDI_SETCALLBACK_DEFN(magAOXMaths, m_indiP_otherVal)(const pcf::IndiProperty &ipRecv)
{
   m_indiP_otherVal = ipRecv;

   updateVals();
   return 0;
}

INDI_NEWCALLBACK_DEFN(magAOXMaths,  m_indiP_setOtherVal)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() ==  m_indiP_setOtherVal.getName())
   {
      std::cerr << " m_indiP_setOtherVal\n";
      
      // received a new value for property val
       m_indiP_setOtherVal["target"] = ipRecv["target"].get<double>();
       m_indiP_setOtherVal.setState (pcf::IndiProperty::Ok);
      //m_indiDriver->sendSetProperty (m_indiP_myVal);

      sendNewProperty(m_indiP_otherVal, "value",  m_indiP_setOtherVal["target"].get<double>());
      
      updateVals();

      return 0;
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //magAOXMaths_hpp
