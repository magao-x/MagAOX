
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
   pcf::IndiProperty x, xmaths;
   pcf::IndiProperty y, ymaths;

   char data[10];
   
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


   INDI_NEWCALLBACK_DECL(magAOXMaths, x);

   INDI_NEWCALLBACK_DECL(magAOXMaths, y);

};

magAOXMaths::magAOXMaths() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void magAOXMaths::setupConfig()
{

}

void magAOXMaths::loadConfig()
{


}

int magAOXMaths::appStartup()
{

   // set up the x input property
   REG_INDI_PROP(x, "x", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);

   x.add (pcf::IndiElement("value"));

   // set up the result maths property
   REG_INDI_PROP_NOCB(xmaths, "xmaths", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   xmaths.add (pcf::IndiElement("value"));
   xmaths.add (pcf::IndiElement("sqr"));
   xmaths.add (pcf::IndiElement("sqrt"));
   xmaths.add (pcf::IndiElement("abs"));

	// set up the x input property
   REG_INDI_PROP(y, "y", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);

   y.add (pcf::IndiElement("value"));

   // set up the result maths property
   REG_INDI_PROP_NOCB(ymaths, "ymaths", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   ymaths.add (pcf::IndiElement("value"));
   ymaths.add (pcf::IndiElement("sqr"));
   ymaths.add (pcf::IndiElement("sqrt"));
   ymaths.add (pcf::IndiElement("abs"));

   
   data[0] = 'a';
   data[1] = 'b';
   for(size_t i=2;i<sizeof(data)-2; ++i) data[i] = ' ';
   data[sizeof(data)-2] = 'z';
   data[sizeof(data)-1] = '\0';
   
   
   log<software_notice>({__FILE__, __LINE__, 1000, 2000, "t"});
   log<software_notice>({__FILE__, __LINE__});
   log<software_warning>({__FILE__, __LINE__});
   log<software_error>({__FILE__, __LINE__});
   log<software_critical>({__FILE__, __LINE__});
   log<software_alert>({__FILE__, __LINE__});
   log<software_emergency>({__FILE__, __LINE__});
   
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

INDI_NEWCALLBACK_DEFN(magAOXMaths, x)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() == x.getName())
   {
      // received a new value for property x

      // extract value
      double v = ipRecv["value"].get<double>();

      // fill maths
      xmaths["value"] = v;
      xmaths["sqr"] = v*v;
      xmaths["sqrt"] = sqrt(v);
      xmaths["abs"] = fabs(v);

      // ack x
      x["value"] = v;
      x.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (x);

      // publish xmaths to be nice
      xmaths.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (xmaths);
      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(magAOXMaths, y)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() == y.getName())
   {
      std::cerr << ipRecv.getDevice() << "\n";
      
      // received a new value for property y

      // extract value
      double v = ipRecv["value"].get<double>();

      // fill maths
      ymaths["value"] = v;
      ymaths["sqr"] = v*v;
      ymaths["sqrt"] = sqrt(v);
      ymaths["abs"] = fabs(v);

      // ack x
      y["value"] = v;
      y.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (y);

      // publish ymaths to be nice
      ymaths.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (ymaths);
      return 0;
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //magAOXMaths_hpp
