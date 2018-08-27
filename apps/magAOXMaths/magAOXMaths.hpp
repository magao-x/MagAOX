
#ifndef magAOXMaths_hpp
#define magAOXMaths_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"



namespace MagAOX
{
namespace app
{

/** MagAO-X application to do math on some numbers
  *
  */
class magAOXMaths : public MagAOXApp<>
{

protected:
   // declare our properties
   pcf::IndiProperty my_val, my_val_maths;
 
   pcf::IndiProperty other_val;

   std::string m_myVal {"x"};
   std::string m_other_devName;
   std::string m_other_valName;
   
   
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

};

magAOXMaths::magAOXMaths() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
  
   
   return;
}

void magAOXMaths::setupConfig()
{
   config.add("my_val", "", "my_val", mx::argType::Required, "", "my_val", false, "string", "The name of this app's value.");
   config.add("other_devName", "", "other_devName", mx::argType::Required, "", "other_devName", false, "string", "The name of the other app name.");
   config.add("other_valName", "", "other_valName", mx::argType::Required, "", "other_valName", false, "string", "The name of the other val name.");
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
   REG_INDI_NEWPROP(my_val, m_myVal, pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);

   my_val.add (pcf::IndiElement("value"));

   // set up the result maths property
   REG_INDI_NEWPROP_NOCB(my_val_maths, "maths", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   my_val_maths.add (pcf::IndiElement("value"));
   my_val_maths.add (pcf::IndiElement("sqr"));
   my_val_maths.add (pcf::IndiElement("sqrt"));
   my_val_maths.add (pcf::IndiElement("abs"));

   REG_INDI_SETPROP(other_val, m_other_devName, m_other_valName);
                    
                    
   
   return 0;
}

int magAOXMaths::appLogic()
{

   //log<text_log>(data);
   //log<loop_closed>();
   return 0;

}

int magAOXMaths::appShutdown()
{

   return 0;
}

INDI_NEWCALLBACK_DEFN(magAOXMaths, my_val)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() == my_val.getName())
   {
      // received a new value for property x

      // extract value
      double v = ipRecv["value"].get<double>();

      // fill maths
      my_val_maths["value"] = v;
      my_val_maths["sqr"] = v*v;
      my_val_maths["sqrt"] = sqrt(v);
      my_val_maths["abs"] = fabs(v);

      // ack x
      my_val["value"] = v;
      my_val.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (my_val);

      // publish xmaths to be nice
      my_val_maths.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (my_val_maths);
      return 0;
   }
   return -1;
}

INDI_SETCALLBACK_DEFN(magAOXMaths, other_val)(const pcf::IndiProperty &ipRecv)
{
   std::cerr << "Got set\n";
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //magAOXMaths_hpp
