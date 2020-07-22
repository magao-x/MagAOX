

#ifndef ttmModulator_hpp
#define ttmModulator_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"


namespace MagAOX
{
namespace app
{

/** MagAO-X application to control TTM modulation
  *
  * \todo need tests fo ttmModulator
  */
class ttmModulator : public MagAOXApp<>
{

protected:

   /** \name Configurable Parameters
     * @{
     */

   double m_maxFreq {3622.0}; ///< The maximum modulation frequency settable by this program
   double m_maxVolt {2.1}; ///< The maximum modulation voltage settable by this program

   double m_setVoltage_1 {5.0}; ///< the set position voltage of Ch. 1.
   double m_setVoltage_2 {5.0}; ///< the set position voltage of Ch. 2.

   double m_setDVolts {1.0}; ///< The setting ramp step size [volts].

   double m_modDFreq {500}; ///< The modulation ramp frequency step size [Hz].
   double m_modDVolts {0.1}; ///< The modulation ramp voltage step size [Volts].

   double m_rotAngle {0};
   double m_rotParity {1};
   
   ///@}

   int m_modState {-1}; ///< -1 = unknown, 0 = off, 1 = rest, 2 = midset, 3 = set, 4 = modulating
   int m_modStateRequested {-1};  ///< The requested TTM state
   double m_modRad {0}; ///< The current modulation radius, in lam/D.
   double m_modRadRequested {-1}; ///< The requested modulation radius, in lam/D.
   double m_modFreq {0}; ///< The current modulation frequency, in Hz.
   double m_modFreqRequested {-1}; ///< The requested modulation frequency, in Hz.


   int m_C1outp {-1};     ///< Output state of fxn gen channel 1.
   double m_C1freq {-1};  ///< Frequency of fxn gen channel 1.
   double m_C1volts {-1}; ///< Voltage p2p of fxn gen channel 1.
   double m_C1ofst {-1};  ///< DC offset of fxn gen channel 1.
   double m_C1phse {-1};  ///< Phase of fxn gen channel 1.

   int m_C2outp {-1};     ///< Output state of fxn gen channel 2
   double m_C2freq {-1};  ///< Frequency of fxn gen channel 2.
   double m_C2volts {-1}; ///< Voltage p2p of fxn gen channel 2.
   double m_C2ofst {-1};  ///< DC offset of fxn gen channel 2.
   double m_C2phse {-1};  ///< Phase of fxn gen channel 2.

   double m_calRadius {3.0};
   
   std::vector<double> m_calFreqs = {100,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000};
   std::vector<double> m_calC1Amps = {0.61,0.61,0.57,0.55,0.53,0.51,0.51,0.49,0.46,0.44,0.44,0.43,0.44,0.46,0.49,0.54,0.58,0.62};
   std::vector<double> m_calC2Amps = {0.6,0.6,0.6,0.57,0.55,0.53,0.51,0.49,0.49,0.49,0.49,0.5,0.52,0.54,0.59,0.64,0.70,0.77};
   std::vector<double> m_calC2Phse = { 75, 75, 75,  75,  75,  75,  75,  75,  75,  75,  75,  75,  75,  72,  72,  70,  70,  70} ;
   
   
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
     * \returns 0 on success
     * \returns -1 on error.
     */
   virtual int appStartup();

   /// Implementation of the FSM for the TTM Modulator
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   virtual int appShutdown();


   /// Calculate the state of the modulator from the fxn gen params.
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   int calcState();

   /// Rest the TTM
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   int restTTM();

   /// Set the TTM
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   int setTTM();

   /// Begin modulating or modify current modulation parameters.
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   int modTTM( double newRad, ///< The new radius for modulation [lam/D]
               double newFreq ///< The new frequency for modulation [Hz]
             );

   int offset12( double d1,
                 double d2
               );
   
   int offsetXY( double dx,
                 double dy
               );
   
protected:

   //declare our properties
   pcf::IndiProperty m_indiP_modState;

   pcf::IndiProperty m_indiP_modRadius;
   pcf::IndiProperty m_indiP_modFrequency;

   pcf::IndiProperty m_indiP_offset12;
   pcf::IndiProperty m_indiP_offset;
   
   
   pcf::IndiProperty m_indiP_FGState;

   pcf::IndiProperty m_indiP_C1outp;
   pcf::IndiProperty m_indiP_C1freq;
   pcf::IndiProperty m_indiP_C1volts;
   pcf::IndiProperty m_indiP_C1ofst;
   pcf::IndiProperty m_indiP_C1phse;

   pcf::IndiProperty m_indiP_C2outp;
   pcf::IndiProperty m_indiP_C2freq;
   pcf::IndiProperty m_indiP_C2volts;
   pcf::IndiProperty m_indiP_C2ofst;
   pcf::IndiProperty m_indiP_C2phse;

public:
   INDI_NEWCALLBACK_DECL(ttmModulator, m_indiP_modState);
   INDI_NEWCALLBACK_DECL(ttmModulator, m_indiP_modRadius);
   INDI_NEWCALLBACK_DECL(ttmModulator, m_indiP_modFrequency);
   INDI_NEWCALLBACK_DECL(ttmModulator, m_indiP_offset12);
   INDI_NEWCALLBACK_DECL(ttmModulator, m_indiP_offset);
   
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1outp);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1freq);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1volts);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1ofst);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C1phse);

   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2outp);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2freq);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2volts);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2ofst);
   INDI_SETCALLBACK_DECL(ttmModulator, m_indiP_C2phse);

};

inline
ttmModulator::ttmModulator() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   return;
}

inline
void ttmModulator::setupConfig()
{
   config.add("limits.maxfreq", "", "limits.maxfreq", argType::Required, "limits", "maxfreq", false, "real", "The maximum frequency [Hz] which can be set through this program.");
   config.add("limits.maxamp", "", "limits.maxamp", argType::Required, "limits", "maxamp", false, "real", "The maximum amplitude [lam/D] which can be set throught this program.");

   config.add("cal.voltsperld1", "", "cal.voltsperld1", argType::Required, "cal", "voltsperld1", false, "real", "The voltage per lam/D for channel 1.");
   config.add("cal.voltsperld2", "", "cal.voltsperld2", argType::Required, "cal", "voltsperld2", false, "real", "The voltage per lam/D for channel 2.");
   config.add("cal.phase", "", "cal.phase", argType::Required, "cal", "phase", false, "real", "The axis phase offset, which is applied to channel 2.");

   config.add("cal.setv1", "", "cal.setv1", argType::Required, "cal", "setv1", false, "real", "The set position voltage of chaannel 1.");
   config.add("cal.setv2", "", "cal.setv2", argType::Required, "cal", "setv2", false, "real", "The set position voltage of chaannel 2.");

   config.add("cal.setDvolts", "", "cal.setDvolts", argType::Required, "cal", "setDvolts", false, "real", "The setting ramp step size [Volts]");

   config.add("cal.modDfreq", "", "cal.modDfreq", argType::Required, "cal", "modDfreq", false, "real", "The modulation ramp frequency step size [Hz]");
   config.add("cal.modDvolts", "", "cal.modDvolts", argType::Required, "cal", "modDvolts", false, "real", "The modulation ramp voltage step size [Volts]");

   config.add("cal.rotAngle", "", "cal.rotAngle", argType::Required, "cal", "rotAngle", false, "real", "The offset rotation matrix angle in degrees.");
   config.add("cal.rotParity", "", "cal.rotParity", argType::Required, "cal", "rotParity", false, "real", "The offset rotation matrix parity, +1 or -1.");
}

inline
void ttmModulator::loadConfig()
{
   config(m_maxFreq, "limits.maxfreq");
   //config(m_maxAmp, "limits.maxamp");
   //config(m_voltsPerLD_1, "cal.voltsperld1");
   //config(m_voltsPerLD_2, "cal.voltsperld2");
   //config(m_phase_2, "cal.phase");

   config(m_setVoltage_1, "cal.setv1");
   config(m_setVoltage_2, "cal.setv2");

   config(m_setDVolts, "cal.setDvolts");
   config(m_modDFreq, "cal.modDfreq");
   config(m_modDVolts, "cal.modDvolts");
   
   config(m_rotAngle, "cal.rotAngle");
   m_rotAngle = m_rotAngle*3.14159/180.;
   
   config(m_rotParity, "cal.rotParity");
   if(m_rotParity < 0) m_rotParity = -1;
   else m_rotParity = 1;
   
   
}

inline
int ttmModulator::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_modState, "modState", pcf::IndiProperty::Number);
   m_indiP_modState.add (pcf::IndiElement("current"));
   m_indiP_modState.add (pcf::IndiElement("requested"));
   m_indiP_modState["current"].set(m_modState);
   m_indiP_modState["requested"].set(m_modStateRequested);

   REG_INDI_NEWPROP(m_indiP_modFrequency, "modFrequency", pcf::IndiProperty::Number);
   m_indiP_modFrequency.add (pcf::IndiElement("current"));
   m_indiP_modFrequency.add (pcf::IndiElement("requested"));
   m_indiP_modFrequency["current"].set(m_modFreq);
   m_indiP_modFrequency["requested"].set(m_modFreqRequested);

   REG_INDI_NEWPROP(m_indiP_modRadius, "modRadius", pcf::IndiProperty::Number);
   m_indiP_modRadius.add (pcf::IndiElement("current"));
   m_indiP_modRadius.add (pcf::IndiElement("requested"));
   m_indiP_modRadius["current"].set(m_modRad);
   m_indiP_modRadius["requested"].set(m_modRadRequested);
  
   REG_INDI_NEWPROP(m_indiP_offset12, "offset12", pcf::IndiProperty::Number);
   m_indiP_offset12.add (pcf::IndiElement("dC1"));
   m_indiP_offset12.add (pcf::IndiElement("dC2"));
   
   REG_INDI_NEWPROP(m_indiP_offset, "offset", pcf::IndiProperty::Number);
   m_indiP_offset.add (pcf::IndiElement("x"));
   m_indiP_offset.add (pcf::IndiElement("y"));
   
   REG_INDI_SETPROP(m_indiP_C1outp, "fxngenmodwfs", "C1outp");
   REG_INDI_SETPROP(m_indiP_C1freq, "fxngenmodwfs", "C1freq");
   REG_INDI_SETPROP(m_indiP_C1volts, "fxngenmodwfs", "C1amp");
   REG_INDI_SETPROP(m_indiP_C1ofst, "fxngenmodwfs", "C1ofst");
   REG_INDI_SETPROP(m_indiP_C1phse, "fxngenmodwfs", "C1phse");

   REG_INDI_SETPROP(m_indiP_C2outp, "fxngenmodwfs", "C2outp");
   REG_INDI_SETPROP(m_indiP_C2freq, "fxngenmodwfs", "C2freq");
   REG_INDI_SETPROP(m_indiP_C2volts, "fxngenmodwfs", "C2amp");
   REG_INDI_SETPROP(m_indiP_C2ofst, "fxngenmodwfs", "C2ofst");
   REG_INDI_SETPROP(m_indiP_C2phse, "fxngenmodwfs", "C2phse");

   return 0;
}

inline
int ttmModulator::appLogic()
{
   if(state()==stateCodes::POWEROFF) return 0;
   
   if(state()==stateCodes::POWERON)
   {
      sleep(2);
   }
   
   if( calcState() < 0 )
   {
      //application failure if we can't determine state
      log<software_critical>({__FILE__,__LINE__});
      return -1;
   }

   if(m_modState == 1 || m_modState == 3)
   {
      state(stateCodes::READY);
      if(!stateLogged()) log<ttmmod_params>({(uint8_t) m_modState, m_modFreq, m_modRad, 0,0});
   }
   if(m_modState == 2)
   {
      state(stateCodes::ERROR);
      if(!stateLogged()) log<ttmmod_params>({(uint8_t) m_modState, m_modFreq, m_modRad, 0,0});
   }
   if(m_modState == 4)
   {
      state(stateCodes::OPERATING);
      if(!stateLogged()) log<ttmmod_params>({(uint8_t) m_modState, m_modFreq, m_modRad, 0,0});
   }

   { //mutex scope
      std::lock_guard<std::mutex> lock(m_indiMutex);
      updateIfChanged(m_indiP_modState, "current", m_modState);
      updateIfChanged(m_indiP_modState, "requested", m_modStateRequested);
      updateIfChanged(m_indiP_modRadius, "current", m_modRad);
      updateIfChanged(m_indiP_modRadius, "requested", m_modRadRequested);
      updateIfChanged(m_indiP_modFrequency, "current", m_modFreq);
      updateIfChanged(m_indiP_modFrequency, "requested", m_modFreqRequested);
   }
   //This is set by an INDI newProperty
   if(m_modStateRequested > 0)
   {
      //Step 0: change the requested state to match, so a new request while we're
      //        processing gets handled.

      std::unique_lock<std::mutex> lock(m_indiMutex);
      int newState = m_modStateRequested;
      double newRad = m_modRadRequested;
      double newFreq = m_modFreqRequested;

      m_modStateRequested = 0;
      m_modFreqRequested = 0;
      m_modRadRequested = 0;

      lock.unlock();

      state(stateCodes::CONFIGURING);
      if(newState == 1) restTTM();
      if(newState == 3) setTTM();
      if(newState == 4) 
      {
         if(newRad <= 0.1 || newFreq <= 1)
         {
            log<text_log>("radius or frequency too low", logPrio::LOG_ERROR);
         }
         else
         {
            modTTM(newRad, newFreq);
         }
      }
      calcState();

      //Do this now for responsiveness.
      if(m_modState == 1 || m_modState == 3)
      {
         state(stateCodes::READY);
         if(!stateLogged()) log<ttmmod_params>({(uint8_t) m_modState, m_modFreq, m_modRad, 0,0});
      }
      if(m_modState == 2)
      {
         state(stateCodes::ERROR);
         if(!stateLogged()) log<ttmmod_params>({(uint8_t) m_modState, m_modFreq, m_modRad, 0,0});
      }
      if(m_modState == 4)
      {
         state(stateCodes::OPERATING);
         if(!stateLogged()) log<ttmmod_params>({(uint8_t) m_modState, m_modFreq, m_modRad, 0,0});
      }

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
      m_modState = 1;
   }
   else if( (m_C1freq == 0 || m_C1volts <= 0.002) && (m_C2freq == 0 || m_C2volts <= 0.002) )
   {
      //To be set:
      // -- sine wave freq is 0 or amp is 0.002
      // -- offset V is at setVoltage
      // -- phase is 0
      if(/*m_C1ofst == m_setVoltage_1 && m_C2ofst == m_setVoltage_2 &&*/ m_C1phse == 0 && m_C2phse == 0 )
      {
         m_modState = 3;
      }
      else
      {
         m_modState = 2; //must be setting
      }
   }
   else
   {
      if(m_C1freq != m_C2freq)
      {
         m_modState = 2;
      }
      else
      {
         //Possibly some more checks
         m_modFreq = m_C1freq;
         
         //Interpolate on C1.
         size_t ngt = 0;
   
         for(ngt = 0; ngt < m_calFreqs.size(); ++ngt)
         {
            if( m_calFreqs[ngt] >= m_modFreq) break;
         }
   
         double terpC1Amp;
         if(ngt == 0 || m_calFreqs[ngt] == m_modFreq)
         {
            terpC1Amp = m_calC1Amps[ngt];
         }
         else
         {
            size_t nlt = ngt -1;
            double dfreq = (m_modFreq - m_calFreqs[nlt])/(m_calFreqs[ngt]-m_calFreqs[nlt]);
            terpC1Amp = m_calC1Amps[nlt] + (m_calC1Amps[ngt]-m_calC1Amps[nlt])*dfreq;
      
         }
   
         m_modRad = m_C1volts/terpC1Amp * m_calRadius;
         
         m_modState = 4;
      }
   }

   return 0;
}

void nanoSleep( unsigned long nsec )
{
   std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>(nsec));
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

      nanoSleep(pauseWait);

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

      nanoSleep(pauseWait);

      clock_gettime(CLOCK_REALTIME, &ts1);
   }

   if(fabs(tgtVal - var) <= tol) return 0;

   std::cerr << "Timeout: " << (ts1.tv_sec - ts0.tv_sec)*1e9 + (ts1.tv_nsec - ts0.tv_nsec) << "\n";
   return -1;

}

inline
int ttmModulator::restTTM()
{
   //Steps:
   //1) Set freqs to 0
   if( sendNewProperty(m_indiP_C1freq, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   if( sendNewProperty(m_indiP_C2freq, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

   //2) Set amps to 0 (really 0.002)
   if( sendNewProperty(m_indiP_C1volts, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   if( sendNewProperty(m_indiP_C2volts, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

   //3) Set phase to 0
   if( sendNewProperty(m_indiP_C1phse, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   if( sendNewProperty(m_indiP_C2phse, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

   //4) Set offset to 0
   if( sendNewProperty(m_indiP_C1ofst, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   if( sendNewProperty(m_indiP_C2ofst, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

   //5) Set outputs to off
   if( sendNewProperty(m_indiP_C1outp, "value", "Off") < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   if( sendNewProperty(m_indiP_C2outp, "value", "Off") < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

   //Now check if values have changed.
   if( waitValue(m_C1freq, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C2freq, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C1volts, 0.002, 1e-10) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C2volts, 0.002,1e-10) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C1phse, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C2phse, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C1ofst, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C2ofst, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C1outp, 0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C2outp, 0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

   log<text_log>("The PyWFS TTM is rested.", logPrio::LOG_NOTICE);

   return 0;
}

inline
int ttmModulator::setTTM()
{
   if(m_modState == 3) //already Set.
   {
      return 0;
   }

   if(m_modState == 4) //Modulating
   {
      log<text_log>("Stopping modulation.", logPrio::LOG_INFO);

      //Steps:
      //1) Set freqs to 0
      if( sendNewProperty(m_indiP_C1freq, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      if( sendNewProperty(m_indiP_C2freq, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      //2) Set amps to 0 (really 0.002)
      if( sendNewProperty(m_indiP_C1volts, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      if( sendNewProperty(m_indiP_C2volts, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      //3) Set phase to 0
      if( sendNewProperty(m_indiP_C1phse, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      if( sendNewProperty(m_indiP_C2phse, "value", 0.0) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      /// \todo should we set the offset here just to be sure?

      //Now check if values have changed.
      if( waitValue(m_C1freq, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
      if( waitValue(m_C2freq, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
      if( waitValue(m_C1volts, 0.002, 1e-10) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
      if( waitValue(m_C2volts, 0.002,1e-10) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
      if( waitValue(m_C1phse, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
      if( waitValue(m_C2phse, 0.0) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

      log<text_log>("PyWFS TTM is set.", logPrio::LOG_NOTICE);
      return 0;
   }

   //Ok, we're in not set or modulating.  Possibly rested, or in a partially set state.

   //Steps:
   //1) Make sure we're fully rested:
   if( m_modState != 1)
   {
      if( restTTM() < 0 ) return log<software_error, -1>({__FILE__, __LINE__});

      sleep(1);
   }

   log<text_log>("Setting the PyWFS TTM.", logPrio::LOG_INFO);

   //2) Set outputs to on
   if( sendNewProperty(m_indiP_C1outp, "value", "On") < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   if( sendNewProperty(m_indiP_C2outp, "value", "On") < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

   if( waitValue(m_C1outp, 1) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
   if( waitValue(m_C2outp, 1) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

   //3) Now we begin ramp . . .
   size_t N1 = m_setVoltage_1/m_setDVolts;
   size_t N2 = m_setVoltage_2/m_setDVolts;

   size_t N = N1;
   if(N2 < N1) N = N2;

   log<text_log>("Ramping with " + std::to_string(N) + " steps. [" + std::to_string(N1) + " " + std::to_string(N2) + "]", logPrio::LOG_DEBUG);

   for(size_t i=1; i< N ; ++i)
   {
      double nv = i*m_setDVolts;

      if(nv < 0 || nv > 10) return log<software_error,-1>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});

      if( sendNewProperty(m_indiP_C1ofst, "value", nv) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      if( waitValue(m_C1ofst, nv, 1e-10) < 0 ) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

      sleep(1);

      if( sendNewProperty(m_indiP_C2ofst, "value", nv) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      if( waitValue(m_C2ofst, nv, 1e-10) < 0 ) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

      sleep(1);
   }

   for(size_t j=N; j< N1;++j)
   {
      double nv = j*m_setDVolts;

      if(nv < 0 || nv > 10) return log<software_error,-1>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});

      if( sendNewProperty(m_indiP_C1ofst, "value", nv) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      if( waitValue(m_C1ofst, nv, 1e-10) < 0 ) return log<software_error, -1>({__FILE__,__LINE__, "fxngen timeout"});

      sleep(1);
   }

   for(size_t j=N; j< N2;++j)
   {
      double nv = j*m_setDVolts;

      if(nv < 0 || nv > 10) return log<software_error,-1>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});

      if( sendNewProperty(m_indiP_C2ofst, "value", nv) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      if( waitValue(m_C2ofst, nv, 1e-10) < 0 ) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

      sleep(1);
   }

   if(m_C1ofst < m_setVoltage_1)
   {
      if( m_setVoltage_1 < 0 ||  m_setVoltage_1 > 10) return log<software_error,-1>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});

      if( (sendNewProperty(m_indiP_C1ofst, "value", m_setVoltage_1) < 0 ) ) return log<software_error,-1>({__FILE__,__LINE__});

      if(waitValue(m_C1ofst, m_setVoltage_1, 1e-10) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

   }

   if(m_C2ofst < m_setVoltage_2)
   {
      if( m_setVoltage_2 < 0 ||  m_setVoltage_2 > 10) return log<software_error,-1>({__FILE__, __LINE__, "Bad voltage calculated.  Refusing."});

      if( (sendNewProperty(m_indiP_C2ofst, "value", m_setVoltage_2) < 0 ) ) return log<software_error,-1>({__FILE__,__LINE__});

      if( waitValue(m_C2ofst, m_setVoltage_2, 1e-10) < 0) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

   }

   log<text_log>("PyWFS TTM is set.", logPrio::LOG_NOTICE);

   return 0;
}

inline
int ttmModulator::modTTM( double newRad,
                          double newFreq
                        )
{
   /// \todo log this
   if(newRad < 0 || newFreq < 0) return 0;

   /// \todo logging in these steps

   //For now: if we enter modulating, we stop modulating.
   /// \todo Implement changing modulation without setting first.
   if( m_modState == 4)
   {
      if(newRad == m_modRad && newFreq == m_modFreq) return 0;

      if( setTTM() < 0 ) return log<software_error, -1>({__FILE__, __LINE__});

      if( calcState() < 0) return log<software_error, -1>({__FILE__,__LINE__});

   }

   //If not set, we first check if we are fully rested.
   if( m_modState < 3 )
   {
      if( setTTM() < 0 ) return log<software_error, -1>({__FILE__, __LINE__});

      if( calcState() < 0 ) return log<software_error, -1>({__FILE__,__LINE__});

      if( m_modState < 3) return log<software_error, -1>({__FILE__,__LINE__, "TTM not set/setable."});
   }

   //Check frequency for safety.
   if(newFreq > m_maxFreq)
   {
      log<text_log>("Requested frequency " + std::to_string(newFreq) + " Hz exceeds limit (" + std::to_string(m_maxFreq) + " Hz). Limiting.", logPrio::LOG_WARNING);
      newFreq = m_maxFreq;
   }

   //Calculate voltage, and normalize and safety-check Parameters
   double voltageC1, voltageC2;

   ///\todo here maximum radius should be frequency dependent.

   
   
   double terpC1Amp = 0;
   double terpC2Amp = 0;
   double terpC2Phse = 0;
   
   size_t ngt = 0;
   
   for(ngt = 0; ngt < m_calFreqs.size(); ++ngt)
   {
      if( m_calFreqs[ngt] >= newFreq) break;
   }
   
   if(ngt == 0 || m_calFreqs[ngt] == newFreq)
   {
      terpC1Amp = m_calC1Amps[ngt];
      terpC2Amp = m_calC2Amps[ngt];
      terpC2Phse = m_calC2Phse[ngt];
   }
   else
   {
      size_t nlt = ngt -1;
      double dfreq = (newFreq - m_calFreqs[nlt])/(m_calFreqs[ngt]-m_calFreqs[nlt]);
      
      terpC1Amp = m_calC1Amps[nlt] + (m_calC1Amps[ngt]-m_calC1Amps[nlt])*dfreq;
      terpC2Amp = m_calC2Amps[nlt] + (m_calC2Amps[ngt]-m_calC2Amps[nlt])*dfreq;
      terpC2Phse = m_calC2Phse[nlt] + (m_calC2Phse[ngt]-m_calC2Phse[nlt])*dfreq;
   }
   

   voltageC1 = terpC1Amp*(newRad/m_calRadius);
   voltageC2 = terpC2Amp*(newRad/m_calRadius); 

   
   if(voltageC1 > m_maxVolt)
   {
      log<text_log>("Requested ch-1 voltge " + std::to_string(voltageC1) + " V exceeds limit (" + std::to_string(m_maxVolt) + " V). Limiting.", logPrio::LOG_WARNING);
      voltageC1 = m_maxVolt;
   }
   
   if(voltageC2 > m_maxVolt)
   {
      log<text_log>("Requested ch-2 voltge " + std::to_string(voltageC2) + " V exceeds limit (" + std::to_string(m_maxVolt) + " V). Limiting.", logPrio::LOG_WARNING);
      voltageC2 = m_maxVolt;
   }
   
   
   //At this point we have safe calibrated voltage for the frequency.

   if( m_modState == 3)
   {
      // 0) set phase
      if( sendNewProperty(m_indiP_C2phse, "value", terpC2Phse) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

      /// \todo should we set the offset here just to be sure?

      //Now check if values have changed.
      if( waitValue(m_C2phse, terpC2Phse, 1e-10) < 0  ) return log<software_error, -1>({__FILE__,__LINE__, "fxngen timeout"});

      // 1) set freq to 100 Hz (or requested if < 100 Hz)
      double nextFreq = 100.0;
      if(nextFreq > newFreq) nextFreq = newFreq;

      //send to device
      if( sendNewProperty(m_indiP_C1freq, "value", nextFreq) < 0 ) log<software_error,-1>({__FILE__,__LINE__});
      if( sendNewProperty(m_indiP_C2freq, "value", nextFreq) < 0 ) log<software_error,-1>({__FILE__,__LINE__});

      //Now check if values have changed.
      if( waitValue(m_C1freq, nextFreq, 1e-10) < 0 ) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
      if( waitValue(m_C2freq, nextFreq, 1e-10) < 0 ) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

      // 2) set amp to 0.1 V (or requested if < 0.1 V)
      double nextVolts1 = 0.1;
      if(nextVolts1 > voltageC1) nextVolts1 = voltageC1;

      double nextVolts2 = 0.1;
      if(nextVolts2 > voltageC2) nextVolts2 = voltageC2;

      //send to device
      if( sendNewProperty(m_indiP_C1volts, "value", nextVolts1) < 0 ) return log<software_error, -1>({__FILE__,__LINE__});
      if( sendNewProperty(m_indiP_C2volts, "value", nextVolts2) < 0 ) return log<software_error, -1>({__FILE__,__LINE__});


      //Now check if values have changed.
      if( waitValue(m_C1volts, nextVolts1, 1e-10) < 0 ) log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
      if( waitValue(m_C2volts, nextVolts2, 1e-10) < 0 ) log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

      // 3) Increase freq to 500 Hz, then in 500 Hz increments
      double currFreq = m_C1freq;

      nextFreq = 500.0;
      if(nextFreq > newFreq) nextFreq = newFreq;

      ///\todo make frequency tolerance a configurable
      while( fabs(currFreq - newFreq) > 1e-4)
      {
         if( sendNewProperty(m_indiP_C1freq, "value", nextFreq) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
         if( sendNewProperty(m_indiP_C2freq, "value", nextFreq) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

         //Now check if values have changed.
         if( waitValue(m_C1freq, nextFreq, 1e-10) < 0 ) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
         if( waitValue(m_C2freq, nextFreq, 1e-10) < 0 ) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

         ///\todo make sleep-time configurable
         sleep(1);
         currFreq = m_C1freq;
         nextFreq = currFreq + 500.0;
         if(nextFreq > newFreq) nextFreq = newFreq;
      }

      // 4) Now increase amplitude in 0.1 V increments.
      double currVolts1 = m_C1volts;
      double currVolts2 = m_C2volts;

      nextVolts1 = 0.2;
      if(nextVolts1 > voltageC1) nextVolts1 = voltageC1;

      nextVolts2 = 0.2;
      if(nextVolts2 > voltageC2) nextVolts2 = voltageC2;
      ///\todo make voltage tolerance a configurable.
      while( fabs(currVolts1 - voltageC1) > 1e-4 || fabs(currVolts2 - voltageC2) > 1e-4)
      {
         if( sendNewProperty(m_indiP_C1volts, "value", nextVolts1) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
         if( sendNewProperty(m_indiP_C2volts, "value", nextVolts2) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});

         //Now check if values have changed.
         if( waitValue(m_C1volts, nextVolts1, 1e-10) < 0 ) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});
         if( waitValue(m_C2volts, nextVolts2, 1e-10) < 0 ) return log<software_error,-1>({__FILE__,__LINE__, "fxngen timeout"});

         sleep(1);
         currVolts1 = m_C1volts;
         nextVolts1 = currVolts1 + 0.1;
         if(nextVolts1 > voltageC1) nextVolts1 = voltageC1;

         currVolts2 = m_C2volts;
         nextVolts2 = currVolts2 + 0.1;
         if(nextVolts2 > voltageC2) nextVolts2 = voltageC2;

      }

      m_modRad = newRad;
      m_modFreq = newFreq;
   }
   else return log<software_error,-1>({__FILE__,__LINE__, "TTM not set but should be by now."});


   return 0;
}

inline
int ttmModulator::offset12( double d1,
                            double d2
                          )
{
   
   if( sendNewProperty(m_indiP_C1ofst, "value", m_C1ofst + d1) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   if( sendNewProperty(m_indiP_C2ofst, "value", m_C2ofst + d2) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   
   return 0;
   
}

inline
int ttmModulator::offsetXY( double dx,
                            double dy
                          )
{
   double cs = cos(m_rotAngle);
   double ss = sin(m_rotAngle);
   
   double rdx = dx * cs - dy * ss;
   double rdy = m_rotParity*(dx * ss + dy * cs);
   
   if( sendNewProperty(m_indiP_C1ofst, "value", m_C1ofst + rdx) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   if( sendNewProperty(m_indiP_C2ofst, "value", m_C2ofst + rdy) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   
   return 0;
   
}

INDI_NEWCALLBACK_DEFN(ttmModulator, m_indiP_modState)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_modState.getName())
   {

      int state = 0;
      try
      {
         state = ipRecv["requested"].get<int>();
      }
      catch(...)
      {
         log<software_error>({__FILE__, __LINE__, "exception caught"});
         return -1;
      }

      m_modStateRequested = state;

      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(ttmModulator, m_indiP_modFrequency)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_modFrequency.getName())
   {

      double nf = -1;
      try
      {
         nf = ipRecv["requested"].get<double>();
      }
      catch(...)
      {
         //do nothing, just means no requested in command.
      }
      if(nf > 0) m_modFreqRequested = nf;

      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(ttmModulator, m_indiP_modRadius)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_modRadius.getName())
   {

      double nr = -1;
      try
      {
         nr = ipRecv["requested"].get<double>();
      }
      catch(...)
      {
         //do nothing, just means no requested in command.
      }
      if(nr > 0) m_modRadRequested = nr;

      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(ttmModulator, m_indiP_offset12)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_offset12.getName())
   {

      double dx = 0;
      if(ipRecv.find("dC1"))
      {
         dx = ipRecv["dC1"].get<double>();
      }
      std::cerr << "dC1: " << dx << "\n";
      
      double dy = 0;
      if(ipRecv.find("dC2"))
      {
         dy = ipRecv["dC2"].get<double>();
      }

      std::cerr << "dC2: " << dy << "\n\n";

      
      return offset12(dx, dy);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(ttmModulator, m_indiP_offset)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_offset.getName())
   {

      double dx = 0;
      if(ipRecv.find("x"))
      {
         dx = ipRecv["x"].get<double>();
      }
      std::cerr << "dx: " << dx << "\n";
      
      double dy = 0;
      if(ipRecv.find("y"))
      {
         dy = ipRecv["y"].get<double>();
      }

      std::cerr << "dy: " << dy << "\n\n";

      
      return offsetXY(dx, dy);
   }
   return -1;
}


INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C1outp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C1outp.getDevice() && ipRecv.getName() == m_indiP_C1outp.getName())
   {
      try
      {
         m_indiP_C1outp = ipRecv;
         std::string outp = ipRecv["value"].getValue();

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
      try
      {
         m_indiP_C1freq = ipRecv;
         double nv = ipRecv["value"].get<double>();

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

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C1volts)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C1volts.getDevice() && ipRecv.getName() == m_indiP_C1volts.getName())
   {
      try
      {
         m_indiP_C1volts = ipRecv;
         double nv = ipRecv["value"].get<double>();

         m_C1volts = nv;
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
      try
      {
         m_indiP_C1ofst = ipRecv;
         double nv = ipRecv["value"].get<double>();

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
      try
      {
         m_indiP_C1phse = ipRecv;
         double nv = ipRecv["value"].get<double>();

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
      try
      {
         m_indiP_C2outp = ipRecv;
         std::string outp = ipRecv["value"].getValue();

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
      try
      {
         m_indiP_C2freq = ipRecv;
         double nv = ipRecv["value"].get<double>();

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

INDI_SETCALLBACK_DEFN(ttmModulator, m_indiP_C2volts)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() == m_indiP_C2volts.getDevice() && ipRecv.getName() == m_indiP_C2volts.getName())
   {
      try
      {
         m_indiP_C2volts = ipRecv;
         double nv = ipRecv["value"].get<double>();

         m_C2volts = nv;
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
      try
      {
         m_indiP_C2ofst = ipRecv;

         double nv = ipRecv["value"].get<double>();

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
      try
      {
         m_indiP_C2phse = ipRecv;
         double nv = ipRecv["value"].get<double>();

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
