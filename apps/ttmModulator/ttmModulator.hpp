

#ifndef ttmModulator_hpp
#define ttmModulator_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"


namespace MagAOX
{
namespace app
{

/** MagAO-X application to control a TTM modulation
  *
  *
  */
class ttmModulator : public MagAOXApp<>
{

protected:

   /** \name Configurable Parameters
     * @{
     */

   double m_maxFreq {1000}; ///< The maximum modulation frequency setable by this program
   double m_maxAmp {6.0}; ///< The maximum modulation amplitude setable by this program

   double m_voltsPerLD_1 {0.87/6.0};
   double m_voltsPerLD_2 {0.87/6.0};
   double m_phase_2 {75.0};

   double m_setVoltage_1 {5.0}; ///< the set position voltage of Ch. 1.
   double m_setVoltage_2 {5.0}; ///< the set position voltage of Ch. 2.

   double m_setDVolts {1.0};

   ///@}

   int m_ttmState {-1}; ///< -1 = unknown, 0 = off, 1 = rest, 2 = midset, 3 = set, 4 = modulating
   int m_ttmStateRequested {-1};

   double m_modRad {0};
   double m_modFreq {0};



   int m_C1outp {-1};
   double m_C1freq {-1};
   double m_C1amp {-1};
   double m_C1ofst {-1};
   double m_C1phse {-1};

   int m_C2outp {-1};
   double m_C2freq {-1};
   double m_C2amp {-1};
   double m_C2ofst {-1};
   double m_C2phse {-1};

public:

   /// Default c'tor.
   ttmModulator();

   /// D'tor, declared and defined for noexcept.
   ~ttmModulator() noexcept
   {}

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Setsup the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for the Siglent SDG
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();


   int calcState();

   int restTTM();

   int setTTM();

   int modTTM();

protected:

   //declare our properties
   pcf::IndiProperty m_indiP_ttmState;

   pcf::IndiProperty m_indiP_modulation;

   pcf::IndiProperty m_indiP_FGState;

   pcf::IndiProperty m_indiP_C1outp;
   pcf::IndiProperty m_indiP_C1freq;
   pcf::IndiProperty m_indiP_C1amp;
   pcf::IndiProperty m_indiP_C1ofst;
   pcf::IndiProperty m_indiP_C1phse;

   pcf::IndiProperty m_indiP_C2outp;
   pcf::IndiProperty m_indiP_C2freq;
   pcf::IndiProperty m_indiP_C2amp;
   pcf::IndiProperty m_indiP_C2ofst;
   pcf::IndiProperty m_indiP_C2phse;



public:
   INDI_NEWCALLBACK_DECL(ttmModulator, m_indiP_ttmState);
   INDI_NEWCALLBACK_DECL(ttmModulator, m_indiP_modulation);

   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1outp);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1freq);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1amp);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1ofst);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1phse);

   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2outp);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2freq);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2amp);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2ofst);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2phse);

};

inline
ttmModulator::ttmModulator() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

inline
void ttmModulator::setupConfig()
{
   config.add("limits.maxfreq", "", "limits.maxfreq", mx::argType::Required, "limits", "maxfreq", false, "real", "The maximum frequency [Hz] which can be set through this program.");
   config.add("limits.maxamp", "", "limits.maxamp", mx::argType::Required, "limits", "maxamp", false, "real", "The maximum amplitude [lam/D] which can be set throught this program.");

   config.add("cal.voltsperld1", "", "cal.voltsperld1", mx::argType::Required, "cal", "voltsperld1", false, "real", "The voltage per lam/D for channel 1.");
   config.add("cal.voltsperld2", "", "cal.voltsperld2", mx::argType::Required, "cal", "voltsperld2", false, "real", "The voltage per lam/D for channel 2.");
   config.add("cal.phase", "", "cal.phase", mx::argType::Required, "cal", "phase", false, "real", "The axis phase offset, which is applied to channel 2.");

   config.add("cal.setv1", "", "cal.setv1", mx::argType::Required, "cal", "setv1", false, "real", "The set position voltage of chaannel 1.");
   config.add("cal.setv2", "", "cal.setv2", mx::argType::Required, "cal", "setv2", false, "real", "The set position voltage of chaannel 2.");


}

inline
void ttmModulator::loadConfig()
{
   config(m_maxFreq, "limits.maxfreq");
   config(m_maxAmp, "limits.maxamp");
   config(m_voltsPerLD_1, "cal.voltsperld1");
   config(m_voltsPerLD_2, "cal.voltsperld2");
   config(m_phase_2, "cal.phase");

   config(m_setVoltage_1, "cal.setv1");
   config(m_setVoltage_2, "call.setv2");

}

inline
int ttmModulator::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_ttmState, "ttmState", pcf::IndiProperty::Number);
   m_indiP_ttmState.add (pcf::IndiElement("value"));
   m_indiP_ttmState["value"].set(m_ttmState);

   REG_INDI_NEWPROP(m_indiP_modulation, "modRad", pcf::IndiProperty::Number);
   m_indiP_modulation.add (pcf::IndiElement("frequency"));
   m_indiP_modulation["frequency"].set(m_modFreq);

   m_indiP_modulation.add (pcf::IndiElement("radius"));
   m_indiP_modulation["radius"].set(m_modRad);

   REG_INDI_SETPROP(m_indiP_C1outp, "ttmfxngen", "C1outp");
   REG_INDI_SETPROP(m_indiP_C1freq, "ttmfxngen", "C1freq");
   REG_INDI_SETPROP(m_indiP_C1amp, "ttmfxngen", "C1amp");
   REG_INDI_SETPROP(m_indiP_C1ofst, "ttmfxngen", "C1ofst");
   REG_INDI_SETPROP(m_indiP_C1phse, "ttmfxngen", "C1phse");

   REG_INDI_SETPROP(m_indiP_C2outp, "ttmfxngen", "C2outp");
   REG_INDI_SETPROP(m_indiP_C2freq, "ttmfxngen", "C2freq");
   REG_INDI_SETPROP(m_indiP_C2amp, "ttmfxngen", "C2amp");
   REG_INDI_SETPROP(m_indiP_C2ofst, "ttmfxngen", "C2ofst");
   REG_INDI_SETPROP(m_indiP_C2phse, "ttmfxngen", "C2phse");

   state(stateCodes::READY);
   return 0;
}

inline
int ttmModulator::appLogic()
{



   int rv = calcState();

   if(rv < 0)
   {
      log<software_critical>({__FILE__,__LINE__});
      return -1;
   }

   updateIfChanged(m_indiP_ttmState, "value", m_ttmState);

   if(m_ttmStateRequested > 0)
   {
      //Step 0: change the requested state to match, so a new request while we're
      //        processing gets handled.

      std::unique_lock<std::mutex> lock(m_indiMutex);
      int newState = m_ttmStateRequested;
      m_ttmStateRequested = 0;

      lock.unlock();

      if(newState == 1) restTTM();
      if(newState == 3) setTTM();


   }
   return 0;

}



inline
int ttmModulator::appShutdown()
{
   //don't bother
   return 0;
}

inline
int ttmModulator::calcState()
{
   //Need TTM power state here.

   if( m_C1outp < 1 || m_C2outp < 1 ) //At least one channel off
   {
      //Need to also check fxn gen pwr state here
      m_ttmState = 1;
   }
   else if( (m_C1freq == 0 || m_C1amp <= 0.002) && (m_C2freq == 0 || m_C2amp <= 0.002) )
   {
      //To be set:
      // -- sine wave freq is 0 or amp is 0.002
      // -- offset V is at setVoltage
      // -- phase is 0
      if(m_C1ofst == m_setVoltage_1 && m_C2ofst == m_setVoltage_2 && m_C1phse == 0 && m_C2phse == 0 )
      {
         m_ttmState = 3;
      }
      else
      {
         m_ttmState = 2; //must be setting
      }
   }
   else
   {
      //Possibly some more checks
      m_ttmState = 4;
   }

   return 0;
}

template<typename T>
int waitValue( const T & var,
               const T & tgtVal,
               unsigned long timeout = 5000000000,
               unsigned long pauseWait = 1000000
              )
{
   if(var == tgtVal) return 0;

   struct timespec ts0, ts1;
   clock_gettime(CLOCK_REALTIME, &ts0);
   ts1 = ts0;


   while( (ts1.tv_sec - ts0.tv_sec)*1e9 + (ts1.tv_nsec - ts0.tv_nsec) < timeout)
   {
      if(var == tgtVal) return 0;

      std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>(pauseWait));
      clock_gettime(CLOCK_REALTIME, &ts1);
   }

   if(var == tgtVal) return 0;

   std::cerr << "Timeout: " << (ts1.tv_sec - ts0.tv_sec)*1e9 + (ts1.tv_nsec - ts0.tv_nsec) << "\n";

   return -1;

}

template<typename T>
int waitValue( const T & var,
               const T & tgtVal,
               double tol,
               unsigned long timeout = 5000000000,
               unsigned long pauseWait = 1000000
              )
{
   if(fabs(tgtVal - var) <= tol) return 0;

   struct timespec ts0, ts1;
   clock_gettime(CLOCK_REALTIME, &ts0);
   ts1 = ts0;

   while( (ts1.tv_sec - ts0.tv_sec)*1e9 + (ts1.tv_nsec - ts0.tv_nsec) < timeout)
   {
      if(fabs(tgtVal - var) <= tol) return 0;

      std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>(pauseWait));
      clock_gettime(CLOCK_REALTIME, &ts1);
   }

   if(fabs(tgtVal - var) <= tol) return 0;

   std::cerr << "Timeout: " << (ts1.tv_sec - ts0.tv_sec)*1e9 + (ts1.tv_nsec - ts0.tv_nsec) << "\n";
   return -1;

}

inline
int ttmModulator::restTTM()
{
   log<text_log>("Resting the PyWFS TTM.", logPrio::LOG_INFO);

   //Steps:
   //1) Set freqs to 0
   if( sendNewProperty(m_indiP_C1freq, "value", 0.0) < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   if( sendNewProperty(m_indiP_C2freq, "value", 0.0) < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   //2) Set amps to 0 (really 0.002)
   if( sendNewProperty(m_indiP_C1amp, "value", 0.0) < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   if( sendNewProperty(m_indiP_C2amp, "value", 0.0) < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   //3) Set phase to 0
   if( sendNewProperty(m_indiP_C1phse, "value", 0.0) < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   if( sendNewProperty(m_indiP_C2phse, "value", 0.0) < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   //4) Set offset to 0
   if( sendNewProperty(m_indiP_C1ofst, "value", 0.0) < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   if( sendNewProperty(m_indiP_C2ofst, "value", 0.0) < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   //5) Set outputs to off
   if( sendNewProperty(m_indiP_C1outp, "value", "Off") < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   if( sendNewProperty(m_indiP_C2outp, "value", "Off") < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   //Now check if values have changed.
   if( (waitValue(m_C1freq, 0.0) < 0) || (waitValue(m_C2freq, 0.0) < 0) ||
        (waitValue(m_C1amp, 0.002, 1e-10) < 0) || (waitValue(m_C2amp, 0.002,1e-10) < 0) ||
         (waitValue(m_C1phse, 0.0) < 0) || (waitValue(m_C2phse, 0.0) < 0) ||
          (waitValue(m_C1ofst, 0.0) < 0) || (waitValue(m_C2ofst, 0.0) < 0) ||
           (waitValue(m_C1outp, 0) < 0) || (waitValue(m_C2outp, 0) < 0) )
   {
      log<software_error>({__FILE__,__LINE__, "fxngen timeout"});
      return -1;
   }

   log<text_log>("The PyWFS TTM is rested.", logPrio::LOG_NOTICE);

   return 0;
}

inline
int ttmModulator::setTTM()
{
   if(m_ttmState == 3) //already Set.
   {
      return 0;
   }

   if(m_ttmState == 4) //Modulating
   {
      log<text_log>("Stopping modulation.", logPrio::LOG_INFO);
      //Steps:
      //1) Set freqs to 0
      if( sendNewProperty(m_indiP_C1freq, "value", 0.0) < 0 )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      if( sendNewProperty(m_indiP_C2freq, "value", 0.0) < 0 )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      //2) Set amps to 0 (really 0.002)
      if( sendNewProperty(m_indiP_C1amp, "value", 0.0) < 0 )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      if( sendNewProperty(m_indiP_C2amp, "value", 0.0) < 0 )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      //3) Set phase to 0
      if( sendNewProperty(m_indiP_C1phse, "value", 0.0) < 0 )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      if( sendNewProperty(m_indiP_C2phse, "value", 0.0) < 0 )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      /// \todo should we set the offset here just to be sure?

      //Now check if values have changed.
      if( (waitValue(m_C1freq, 0.0) < 0) || (waitValue(m_C2freq, 0.0) < 0) ||
           (waitValue(m_C1amp, 0.002, 1e-10) < 0) || (waitValue(m_C2amp, 0.002,1e-10) < 0) ||
            (waitValue(m_C1phse, 0.0) < 0) || (waitValue(m_C2phse, 0.0) < 0)  )
      {
         log<software_error>({__FILE__,__LINE__, "fxngen timeout"});
         return -1;
      }

      log<text_log>("PyWFS TTM is set.", logPrio::LOG_NOTICE);
      return 0;
   }

   //Ok, we're in not set or modulating.  Possibly rested, or in a partially set state.

   //Steps:
   //1) Make sure we're fully rested:
   if( m_ttmState != 1)
   {
      if( restTTM() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return -1;
      }
      sleep(1);
   }


   log<text_log>("Setting the PyWFS TTM.", logPrio::LOG_INFO);

   //2) Set outputs to on
   if( (sendNewProperty(m_indiP_C1outp, "value", "On") < 0 ) ||
        ( sendNewProperty(m_indiP_C2outp, "value", "On") < 0 ) )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   };

   if( (waitValue(m_C1outp, 1) < 0) || (waitValue(m_C2outp, 1) < 0) )
   {
      log<software_error>({__FILE__,__LINE__, "fxngen timeout"});
      return -1;
   }

   //3) Now we begin ramp . . .
   size_t N1 = m_setVoltage_1/m_setDVolts;
   size_t N2 = m_setVoltage_2/m_setDVolts;

   size_t N = N1;
   if(N2 < N1) N = N2;

   log<text_log>("Ramping with " + std::to_string(N) + " steps. [" + std::to_string(N1) + " " + std::to_string(N2) + "]", logPrio::LOG_DEBUG);

   for(size_t i=1; i< N ; ++i)
   {
      double nv = i*m_setDVolts;

      if(nv < 0 || nv > 10)
      {
         log<software_error>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});
         return -1;
      }

      if( sendNewProperty(m_indiP_C1ofst, "value", nv) < 0 )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      if( waitValue(m_C1ofst, nv, 1e-10) < 0 )
      {
         log<software_error>({__FILE__,__LINE__, "fxngen timeout"});
         return -1;
      }

      sleep(1);

      if( sendNewProperty(m_indiP_C2ofst, "value", nv) < 0 )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      if( waitValue(m_C2ofst, nv, 1e-10) < 0 )
      {
         log<software_error>({__FILE__,__LINE__, "fxngen timeout"});
         return -1;
      }

      sleep(1);
   }

   for(size_t j=N; j< N1;++j)
   {
      double nv = j*m_setDVolts;

      if(nv < 0 || nv > 10)
      {
         log<software_error>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});
         return -1;
      }

      if( (sendNewProperty(m_indiP_C1ofst, "value", nv) < 0 ) )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      if( (waitValue(m_C1ofst, nv, 1e-10) < 0) )
      {
         log<software_error>({__FILE__,__LINE__, "fxngen timeout"});
         return -1;
      }

      sleep(1);
   }

   for(size_t j=N; j< N2;++j)
   {
      double nv = j*m_setDVolts;

      if(nv < 0 || nv > 10)
      {
         log<software_error>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});
         return -1;
      }

      if( (sendNewProperty(m_indiP_C2ofst, "value", nv) < 0 ) )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      if( (waitValue(m_C2ofst, nv, 1e-10) < 0) )
      {
         log<software_error>({__FILE__,__LINE__, "fxngen timeout"});
         return -1;
      }

      sleep(1);
   }

   if(m_C1ofst < m_setVoltage_1)
   {
      if( m_setVoltage_1 < 0 ||  m_setVoltage_1 > 10)
      {
         log<software_error>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});
         return -1;
      }

      if( (sendNewProperty(m_indiP_C1ofst, "value", m_setVoltage_1) < 0 ) )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      if(waitValue(m_C1ofst, m_setVoltage_1, 1e-10) < 0)
      {
         log<software_error>({__FILE__,__LINE__, "fxngen timeout"});
         return -1;
      }
   }

   if(m_C2ofst < m_setVoltage_2)
   {
      if( m_setVoltage_2 < 0 ||  m_setVoltage_2 > 10)
      {
         log<software_error>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});
         return -1;
      }

      if( (sendNewProperty(m_indiP_C2ofst, "value", m_setVoltage_2) < 0 ) )
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      };

      if(waitValue(m_C2ofst, m_setVoltage_2, 1e-10) < 0)
      {
         log<software_error>({__FILE__,__LINE__, "fxngen timeout"});
         return -1;
      }
   }

   log<text_log>("PyWFS TTM is set.", logPrio::LOG_NOTICE);

   return 0;
}

inline
int ttmModulator::modTTM()
{
   return 0;
}


INDI_NEWCALLBACK_DEFN(ttmModulator, m_indiP_ttmState)(const pcf::IndiProperty &ipRecv)
{
   std::cerr << "State change request" << std::endl;
   if (ipRecv.getName() == m_indiP_ttmState.getName())
   {

      //std::lock_guard<std::mutex> lock(m_indiMutex);

      int state = 0;
      try
      {
         state = ipRecv["value"].get<int>();
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception caught"});
         return -1;
      }

      m_ttmStateRequested = state;

      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(ttmModulator, m_indiP_modulation)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_modulation.getName())
   {


      return 0;
   }
   return -1;
}



INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C1outp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C1outp.getDevice() && ipRecv.getName() == m_indiP_C1outp.getName())
   {
      std::string outp;
      try
      {
         m_indiP_C1outp = ipRecv;
         outp = ipRecv["value"].getValue();

         if( outp == "Off" )
         {
            m_C1outp = 0;
         }
         else if (outp == "On")
         {
            m_C1outp = 1;
         }
         else
         {
            m_C1outp = -1;
         }

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C1freq)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C1freq.getDevice() && ipRecv.getName() == m_indiP_C1freq.getName())
   {
      double nv;
      try
      {
         m_indiP_C1freq = ipRecv;
         nv = ipRecv["value"].get<double>();

         m_C1freq = nv;

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C1amp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C1amp.getDevice() && ipRecv.getName() == m_indiP_C1amp.getName())
   {
      double nv;
      try
      {
         m_indiP_C1amp = ipRecv;
         nv = ipRecv["value"].get<double>();

         m_C1amp = nv;

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C1ofst)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C1ofst.getDevice() && ipRecv.getName() == m_indiP_C1ofst.getName())
   {
      double nv;
      try
      {
         m_indiP_C1ofst = ipRecv;
         nv = ipRecv["value"].get<double>();

         m_C1ofst = nv;

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C1phse)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C1phse.getDevice() && ipRecv.getName() == m_indiP_C1phse.getName())
   {
      double nv;
      try
      {
         m_indiP_C1phse = ipRecv;
         nv = ipRecv["value"].get<double>();

         m_C1phse = nv;

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C2outp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C2outp.getDevice() && ipRecv.getName() == m_indiP_C2outp.getName())
   {
      std::string outp;
      try
      {
         m_indiP_C2outp = ipRecv;
         outp = ipRecv["value"].getValue();

         if( outp == "Off" )
         {
            m_C2outp = 0;
         }
         else if (outp == "On")
         {
            m_C2outp = 1;
         }
         else
         {
            m_C2outp = -1;
         }

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C2freq)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C2freq.getDevice() && ipRecv.getName() == m_indiP_C2freq.getName())
   {
      double nv;
      try
      {
         m_indiP_C2freq = ipRecv;
         nv = ipRecv["value"].get<double>();

         m_C2freq = nv;

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C2amp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C2amp.getDevice() && ipRecv.getName() == m_indiP_C2amp.getName())
   {
      double nv;
      try
      {
         m_indiP_C2amp = ipRecv;
         nv = ipRecv["value"].get<double>();

         m_C2amp = nv;

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C2ofst)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C2ofst.getDevice() && ipRecv.getName() == m_indiP_C2ofst.getName())
   {
      double nv;
      try
      {
         m_indiP_C2ofst = ipRecv;

         nv = ipRecv["value"].get<double>();

         m_C2ofst = nv;

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C2phse)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C2phse.getDevice() && ipRecv.getName() == m_indiP_C2phse.getName())
   {
      double nv;
      try
      {
         m_indiP_C2phse = ipRecv;
         nv = ipRecv["value"].get<double>();

         m_C2phse = nv;

         return 0;
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception from libcommon"});
         return -1;
      }
   }
   log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //ttmModulator_hpp
