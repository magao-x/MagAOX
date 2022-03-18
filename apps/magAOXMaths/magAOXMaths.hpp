
#ifndef magAOXMaths_hpp
#define magAOXMaths_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <limits>

namespace MagAOX
{
namespace app
{

void externalLog ( const std::string & name,
                         const int & code,
                         const std::string & valueStr,
                         const std::string & source
                       )
{
   std::cerr << name << " " << code << " " << valueStr << " " << source << "\n";
}


/** MagAO-X application to do math on some numbers
  *
  */
class magAOXMaths : public MagAOXApp<>
{

protected:
   // declare our properties
   pcf::IndiProperty my_val, my_val_maths;

   pcf::IndiProperty other_val;

   pcf::IndiProperty set_other_val;
   
   std::string m_myVal {"x"};
   std::string m_other_devName;
   std::string m_other_valName;

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


   INDI_NEWCALLBACK_DECL(magAOXMaths, my_val);

   INDI_SETCALLBACK_DECL(magAOXMaths, other_val);
   
   INDI_NEWCALLBACK_DECL(magAOXMaths, set_other_val);

};

magAOXMaths::magAOXMaths() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void magAOXMaths::setupConfig()
{
   config.add("my_val", "", "my_val", argType::Required, "", "my_val", false, "string", "The name of this app's value.");
   config.add("other_devName", "", "other_devName", argType::Required, "", "other_devName", false, "string", "The name of the other app name.");
   config.add("other_valName", "", "other_valName", argType::Required, "", "other_valName", false, "string", "The name of the other val name.");
}

void magAOXMaths::loadConfig()
{
   config(m_myVal, "my_val");
   config(m_other_devName, "other_devName");
   config(m_other_valName, "other_valName");

}

int magAOXMaths::appStartup()
{
   // set up the x input property
   REG_INDI_NEWPROP(my_val, m_myVal, pcf::IndiProperty::Number);
   indi::addNumberElement<double>( my_val, "value",  std::numeric_limits<double>::min(),  std::numeric_limits<double>::max(), 1.0,  "%f", "");
   my_val["value"].set<double>(0.0);


   // set up the result maths property
   REG_INDI_NEWPROP_NOCB(my_val_maths, "maths", pcf::IndiProperty::Number);
   indi::addNumberElement<double>(my_val_maths,"value", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");
   indi::addNumberElement<double>(my_val_maths, "sqr", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");
   indi::addNumberElement<double>(my_val_maths, "sqrt", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");
   indi::addNumberElement<double>(my_val_maths, "abs", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");
   indi::addNumberElement<double>(my_val_maths, "prod", std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0, "%f", "");

   REG_INDI_SETPROP(other_val, m_other_devName, m_other_valName);
   other_val.add (pcf::IndiElement("value"));
   other_val["value"].set<double>(0.0);

   REG_INDI_NEWPROP(set_other_val, "other_val", pcf::IndiProperty::Number);
   indi::addNumberElement<double>( set_other_val, "value",  std::numeric_limits<double>::min(),  std::numeric_limits<double>::max(), 1.0,  "%f", "");
   set_other_val["value"].set<double>(0.0);
   
   
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
   double v = my_val["value"].get<double>();

   if(v == -1) log<text_log>( "value set to -1!", logPrio::LOG_WARNING);
   if(v == -2) log<text_log>( "value set to -2!", logPrio::LOG_ERROR);
   if(v == -3) log<text_log>( "value set to -3!", logPrio::LOG_CRITICAL);
   if(v == -4) log<text_log>( "value set to -4!", logPrio::LOG_ALERT);
   if(v == -5) log<text_log>( "value set to -5!", logPrio::LOG_EMERGENCY);
   
   // fill maths
   my_val_maths["value"] = v;
   my_val_maths["sqr"] = v*v;
   my_val_maths["sqrt"] = sqrt(v);
   my_val_maths["abs"] = fabs(v);

   my_val_maths["prod"] = v*other_val["value"].get<double>();

   log<text_log>("set new value: " + std::to_string(v), logPrio::LOG_NOTICE);
   // publish maths
   my_val_maths.setState (pcf::IndiProperty::Ok);
   if(m_indiDriver) m_indiDriver->sendSetProperty (my_val_maths);

   return 0;
}

INDI_NEWCALLBACK_DEFN(magAOXMaths, my_val)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() == my_val.getName())
   {
      // received a new value for property val
      my_val["value"] = ipRecv["value"].get<double>();
      my_val.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (my_val);

      updateVals();

      return 0;
   }
   return -1;
}

INDI_SETCALLBACK_DEFN(magAOXMaths, other_val)(const pcf::IndiProperty &ipRecv)
{
   other_val = ipRecv;

   updateVals();
   return 0;
}

INDI_NEWCALLBACK_DEFN(magAOXMaths, set_other_val)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() == set_other_val.getName())
   {
      std::cerr << "set_other_val\n";
      
      // received a new value for property val
      set_other_val["value"] = ipRecv["value"].get<double>();
      set_other_val.setState (pcf::IndiProperty::Ok);
      //m_indiDriver->sendSetProperty (my_val);

      sendNewProperty(other_val, "value", set_other_val["value"].get<double>());
      
      //updateVals();

      return 0;
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //magAOXMaths_hpp
