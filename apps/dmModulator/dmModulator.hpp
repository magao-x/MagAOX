/** \file dmModulator.hpp
  * \brief The MagAO-X DM speckle maker header file
  *
  * \ingroup dmModulator_files
  */

#ifndef dmModulator_hpp
#define dmModulator_hpp

#include <mx/improc/eigenCube.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/ioutils/stringUtils.hpp>
#include <mx/sys/timeUtils.hpp>
#include <mx/sigproc/fourierModes.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/math/randomT.hpp>

/** \defgroup dmModulator
  * \brief The DM speckle maker app
  * 
  * Creates a set of fourier modes to generate speckles, then applies them to a DM channel
  * at a specified rate.
  * 
  *
  * <a href="../handbook/operating/software/apps/dmModulator.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup dmModulator_files
  * \ingroup dmModulator
  */


namespace MagAOX
{
namespace app
{

/// The MagAO-X DM mode commander
/** 
  * \ingroup dmModulator
  */
class dmModulator : public MagAOXApp<true>
{

   typedef float realT;
   
   friend class dmModulator_test;

protected:

   /** \name Configurable Parameters
     *@{
     */

   std::string m_shapeCube;
      
   std::string m_dmName; ///< The descriptive name of this dm. Default is the channel name.
   
   std::string m_dmChannelName; ///< The name of the DM channel to write to.
   
   std::string m_dmTriggerChannel; ///< The DM channel to monitor as a trigger

   int m_triggerSemaphore {9}; ///< The semaphore to use (default 9)

   bool m_trigger {true}; ///< Run in trigger mode if true (default)

   realT m_amp {0.01}; ///< The speckle amplitude on the DM

   realT m_frequency {2000}; ///< The frequency to modulate at if not triggering (default 2000 Hz)

   bool m_noise = true;
   ///@}


   mx::math::normDistT<realT> m_normDist;

   mx::improc::eigenCube<realT> m_shapes;
      
   IMAGE m_imageStream; 
   uint32_t m_width {0}; ///< The width of the image
   uint32_t m_height {0}; ///< The height of the image.
   
   IMAGE m_triggerStream;

   uint8_t m_dataType{0}; ///< The ImageStreamIO type code.
   size_t m_typeSize {0}; ///< The size of the type, in bytes.  
   
   bool m_opened {true};
   bool m_restart {false};
   
   bool m_modulating {false};

public:
   /// Default c'tor.
   dmModulator();

   /// D'tor, declared and defined for noexcept.
   ~dmModulator() noexcept
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

   /// Implementation of the FSM for dmModulator.
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

protected:

   /** \name Modulator Thread 
     * This thread sends the signal to the dm at the prescribed frequency
     *
     * @{
     */ 
   int m_modThreadPrio {60}; ///< Priority of the modulator thread, should normally be > 00.

   std::string m_modThreadCpuset; ///< The cpuset for the modulator thread.

   std::thread m_modThread; ///< A separate thread for the modulation

   bool m_modThreadInit {true}; ///< Synchronizer to ensure f.g. thread initializes before doing dangerous things.
   
   pid_t m_modThreadID {0}; ///< Modulate thread PID.

   pcf::IndiProperty m_modThreadProp; ///< The property to hold the modulator thread details.

   ///Thread starter, called by modThreadStart on thread construction.  Calls modThreadExec.
   static void modThreadStart( dmModulator * d /**< [in] a pointer to a dmModulator instance (normally this) */);

   /// Execute the frame grabber main loop.
   void modThreadExec();

   ///@}

   //INDI:
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_dm;
   pcf::IndiProperty m_indiP_amp;
   pcf::IndiProperty m_indiP_frequency;
   pcf::IndiProperty m_indiP_trigger;
   pcf::IndiProperty m_indiP_modulating;
   pcf::IndiProperty m_indiP_zero;

   std::vector<std::string> m_elNames;
public:
   INDI_NEWCALLBACK_DECL(dmModulator, m_indiP_amp);
   INDI_NEWCALLBACK_DECL(dmModulator, m_indiP_frequency);
   INDI_NEWCALLBACK_DECL(dmModulator, m_indiP_trigger);
   INDI_NEWCALLBACK_DECL(dmModulator, m_indiP_modulating);
   INDI_NEWCALLBACK_DECL(dmModulator, m_indiP_zero);
   
};

dmModulator::dmModulator() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void dmModulator::setupConfig()
{
   config.add("dm.shapeCube", "", "dm.shapeCube", argType::Required, "dm", "shapeCube", false, "string", "Full path to the FITS file containing the shapes to modulate on this DM.");
   config.add("dm.name", "", "dm.name", argType::Required, "dm", "name", false, "string", "The descriptive name of this dm. Default is the channel name.");
   config.add("dm.channelName", "", "dm.channelName", argType::Required, "dm", "channelName", false, "string", "The name of the DM channel to write to.");
   config.add("dm.triggerChannel", "", "dm.triggerChannel", argType::Required, "dm", "triggerChannel", false, "string", "The name of the DM channel to trigger on.");
   config.add("dm.triggerSemaphore", "", "dm.triggerSemaphore", argType::Required, "dm", "triggerSemaphore", false, "int", "The semaphore to use (default 9).");
   config.add("dm.trigger", "", "dm.trigger", argType::True, "dm", "trigger", false, "bool", "Run in trigger mode if true (default).");
   config.add("dm.amp", "", "dm.amp", argType::Required, "dm", "amp", false, "float", "The speckle amplitude on the DM (default 0.01).");

   config.add("dm.frequency", "", "dm.frequency", argType::Required, "dm", "frequency", false, "float", "The frequency to modulate at if not triggering (default 2000 Hz).");

   config.add("modulator.threadPrio", "", "modulator.threadPrio", argType::Required, "modulator", "threadPrio", false, "int", "The real-time priority of the modulator thread.");

   config.add("modulator.cpuset", "", "modulator.cpuset", argType::Required, "modulator", "cpuset", false, "string", "The cpuset to assign the modulator thread to.");

}

int dmModulator::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_dmChannelName, "dm.channelName");
   
   m_dmName = m_dmChannelName;
   _config(m_dmName, "dm.name");

   _config(m_dmTriggerChannel, "dm.triggerChannel");
   _config(m_triggerSemaphore, "dm.triggerSemaphore");
   if(_config.isSet("dm.trigger")) _config(m_trigger, "dm.trigger");
   _config(m_amp, "dm.amp");
   _config(m_frequency, "dm.frequency");
   _config(m_frequency, "dm.frequency");
   _config(m_shapeCube,"dm.shapeCube" );
   _config(m_modThreadPrio, "modulator.threadPrio");
   _config(m_modThreadCpuset, "modulator.cpuset");
   
   return 0;
}

void dmModulator::loadConfig()
{
   loadConfigImpl(config);
}

int dmModulator::appStartup()
{
   mx::fits::fitsFile<realT> ff;
   
   if(ff.read(m_shapes, m_shapeCube) < 0) 
   {
      return log<text_log,-1>("Could not open mode cube file", logPrio::LOG_ERROR);
   }

   REG_INDI_NEWPROP_NOCB(m_indiP_dm, "dm", pcf::IndiProperty::Text);
   m_indiP_dm.add(pcf::IndiElement("name"));
   m_indiP_dm["name"] = m_dmName;
   m_indiP_dm.add(pcf::IndiElement("channel"));
   m_indiP_dm["channel"] = m_dmChannelName;
   
   createStandardIndiNumber<float>(m_indiP_amp, "amp", -1, 0,1, "%f");
   m_indiP_amp["current"] = m_amp;
   m_indiP_amp["target"] = m_amp;
   registerIndiPropertyNew( m_indiP_amp, INDI_NEWCALLBACK(m_indiP_amp));

   createStandardIndiNumber<float>(m_indiP_frequency, "frequency", 0, 0,10000, "%f");
   m_indiP_frequency["current"] = m_frequency;
   m_indiP_frequency["target"] = m_frequency;
   registerIndiPropertyNew( m_indiP_frequency, INDI_NEWCALLBACK(m_indiP_frequency));
   
   createStandardIndiToggleSw( m_indiP_trigger, "trigger");
   if( registerIndiPropertyNew( m_indiP_trigger, INDI_NEWCALLBACK(m_indiP_trigger)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   if(m_trigger)
   {
      m_indiP_trigger["toggle"] = pcf::IndiElement::On;
   }
   else
   {
      m_indiP_trigger["toggle"] = pcf::IndiElement::Off;
   }

   createStandardIndiToggleSw( m_indiP_modulating, "modulating");
   if( registerIndiPropertyNew( m_indiP_modulating, INDI_NEWCALLBACK(m_indiP_modulating)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiRequestSw( m_indiP_zero, "zero");
   if( registerIndiPropertyNew( m_indiP_zero, INDI_NEWCALLBACK(m_indiP_zero)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   if(threadStart( m_modThread, m_modThreadInit, m_modThreadID, m_modThreadProp, m_modThreadPrio, m_modThreadCpuset,"modulator", this, modThreadStart)  < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
      
   state(stateCodes::NOTCONNECTED);
   
   
   return 0;
}

int dmModulator::appLogic()
{
   if(state() == stateCodes::NOTCONNECTED)
   {
      m_opened = false;
      m_restart = false; //Set this up front, since we're about to restart.
      
      if( ImageStreamIO_openIm(&m_imageStream, m_dmChannelName.c_str()) == 0)
      {
         if(m_imageStream.md[0].sem < 10) ///<\todo this is hardcoded in ImageStreamIO.c -- should be a define
         {
            ImageStreamIO_closeIm(&m_imageStream);
         }
         else
         {
            m_opened = true;
         }
      }
      
      //Only bother to try if previous worked and we have a spec
      if(m_opened == true && m_dmTriggerChannel != "")
      {
         if( ImageStreamIO_openIm(&m_triggerStream, m_dmTriggerChannel.c_str()) == 0)
         {
            if(m_triggerStream.md[0].sem < 10) ///<\todo this is hardcoded in ImageStreamIO.c -- should be a define
            {
               ImageStreamIO_closeIm(&m_triggerStream);
               m_opened = false;
            }
         }
      }

      if(m_opened)
      {
         state(stateCodes::CONNECTED);
      }
   }
   
   if(state() == stateCodes::CONNECTED)
   {
      m_dataType = m_imageStream.md[0].datatype;
      m_typeSize = ImageStreamIO_typesize(m_dataType);
      m_width = m_imageStream.md[0].size[0];
      m_height = m_imageStream.md[0].size[1];
   
   
      if(m_dataType != _DATATYPE_FLOAT )
      {
         return log<text_log,-1>("Data type of DM channel is not float.", logPrio::LOG_CRITICAL);
      }
   
      if(m_typeSize != sizeof(realT))
      {
         return log<text_log,-1>("Type-size mismatch, realT is not float.", logPrio::LOG_CRITICAL);
      }

      state(stateCodes::READY);
   }
   
   return 0;
}

int dmModulator::appShutdown()
{
   if(m_modThread.joinable())
   {
      try
      {
         m_modThread.join(); //this will throw if it was already joined
      }
      catch(...)
      {
      }
   }

   return 0;
}


inline
void dmModulator::modThreadStart( dmModulator * d)
{
   d->modThreadExec();
}

inline
void dmModulator::modThreadExec()
{
   m_modThreadID = syscall(SYS_gettid);

   mx::improc::eigenCube<realT> ampShapes;

   mx::improc::eigenImage<realT> noise;

   //Wait fpr the thread starter to finish initializing this thread.
   while( (m_modThreadInit == true || state() != stateCodes::READY) && m_shutdown == 0)
   {
      sleep(1);
   }
   
   while(m_shutdown == 0)
   {
      if(!m_modulating && !m_shutdown) //If we aren't modulating we sleep for 1/2 a second
      {
         mx::sys::milliSleep(500);
      }
      
      if(m_modulating && !m_shutdown)
      {
         
         int64_t freqNsec = (1.0/m_frequency)*1e9;
         updateIfChanged(m_indiP_frequency, "current", m_frequency);
         int64_t dnsec;
   
         int idx = 0;
         
         timespec modstart;
         timespec currtime;
      
         bool triggered = false;
         sem_t * sem = nullptr;
         if(m_dmTriggerChannel == "") 
         {
            m_trigger = false;
            indi::updateSwitchIfChanged(m_indiP_trigger, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
         }
         else if(m_trigger == true)
         {
            ImageStreamIO_semflush(&m_triggerStream, m_triggerSemaphore);
      
            sem = m_triggerStream.semptr[m_triggerSemaphore]; ///< The semaphore to monitor for new image data
         }

         ampShapes.resize(m_shapes.rows(), m_shapes.cols(), m_shapes.planes());
         noise.resize(m_width, m_height);
         for(int cc=0;cc<noise.cols(); ++cc)
         {
            for(int rr=0;rr<noise.rows(); ++rr)
            {
               noise(rr,cc) = m_normDist*m_amp;
            }
         }

         for(int p =0; p < ampShapes.planes(); ++p)
         {
            ampShapes.image(p) = m_shapes.image(p) * m_amp;
         }
         updateIfChanged(m_indiP_amp, "current", m_amp);

         log<text_log>("started modulating",logPrio::LOG_NOTICE);
         
         dnsec = freqNsec;
         clock_gettime(CLOCK_REALTIME, &modstart);

         while(m_modulating && !m_shutdown)
         {
            if(m_trigger)
            {
               timespec ts;

               if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
               {
                  log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); 
                  return;
               }
         
               ts.tv_sec += 1;
         
               if(sem_timedwait(sem, &ts) == 0)
               {
                  triggered = true;
               }
               else
               {
                  triggered = false;

                  //Check for why we timed out
                  if(errno == EINTR) break; //This indicates signal interrupted us, time to restart or shutdown, loop will exit normally if flags set.
            
                  //ETIMEDOUT just means we should wait more.
                  //Otherwise, report an error.
                  if(errno != ETIMEDOUT)
                  {
                     log<software_error>({__FILE__, __LINE__,errno, "sem_timedwait"});
                     break;
                  }
               }
            }
            else
            {
               mx::sys::nanoSleep(0.5*dnsec);
               clock_gettime(CLOCK_REALTIME, &currtime);
         
               dnsec = (currtime.tv_sec - modstart.tv_sec)*1000000000 + (currtime.tv_nsec - modstart.tv_nsec);
               triggered = false;
            }

            if(dnsec >= freqNsec || triggered)
            {
               //Do the write
               
               m_imageStream.md->write = 1;
   
               if(m_noise)
               {
                  memcpy(m_imageStream.array.raw, noise.data(), m_width*m_height*m_typeSize);
               }
               else
               {   
                  memcpy(m_imageStream.array.raw, ampShapes.image(idx).data(), m_width*m_height*m_typeSize);
               }

               m_imageStream.md->atime = currtime;
               m_imageStream.md->writetime = currtime;
               
               if(!m_trigger) m_imageStream.md->cnt0++;
   
               m_imageStream.md->write=0;
               ImageStreamIO_sempost(&m_imageStream,-1);
   
               ++idx;
               if(idx >= m_shapes.planes()) idx=0;
               
               if(m_noise)
               {
                  for(int cc=0;cc<noise.cols(); ++cc)
                  {
                     for(int rr=0;rr<noise.rows(); ++rr)
                     {
                        noise(rr,cc) = m_normDist*m_amp;
                     }
                  }
               }

               if(!m_trigger)
               {
                  modstart.tv_nsec += freqNsec;
                  if(modstart.tv_nsec >= 1000000000)
                  {
                     modstart.tv_nsec -= 1000000000;
                     modstart.tv_sec += 1;
                  }
                  dnsec = freqNsec;
               }
            }
         }
         log<text_log>("stopped modulating", logPrio::LOG_NOTICE);
         //Always zero when done
         clock_gettime(CLOCK_REALTIME, &currtime);
         m_imageStream.md->write = 1;
   
         memset(m_imageStream.array.raw, 0.0, m_width*m_height*m_typeSize);

         m_imageStream.md->atime = currtime;
         m_imageStream.md->writetime = currtime;
               
         if(!m_trigger) m_imageStream.md->cnt0++;
   
         m_imageStream.md->write=0;
         ImageStreamIO_sempost(&m_imageStream,-1);
         log<text_log>("zeroed");

      }
   }
   
}

INDI_NEWCALLBACK_DEFN(dmModulator, m_indiP_amp)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_amp.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }

   float amp = -1000000000;
   
   if( ipRecv.find("current") )
   {
      amp = ipRecv["current"].get<float>();
   }
   
   if( ipRecv.find("target") )
   {
      amp = ipRecv["target"].get<float>();
   }

   if(amp == -1000000000)
   {
      log<software_error>({__FILE__,__LINE__, "Invalid requested amp: " + std::to_string(amp)});
      return 0;
   }
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   m_amp = amp;
   updateIfChanged(m_indiP_amp, "target", m_amp);
   return 0;
}

INDI_NEWCALLBACK_DEFN(dmModulator, m_indiP_frequency)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_frequency.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }

   float freq = -1;
   
   if( ipRecv.find("current") )
   {
      freq = ipRecv["current"].get<float>();
   }
   
   if( ipRecv.find("target") )
   {
      freq = ipRecv["target"].get<float>();
   }

   if(freq < 0)
   {
      log<software_error>({__FILE__,__LINE__, "Invalid requested frequency: " + std::to_string(freq)});
      return 0;
   }
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   m_frequency = freq;
   updateIfChanged(m_indiP_frequency, "target", m_frequency);
   return 0;
}

INDI_NEWCALLBACK_DEFN(dmModulator, m_indiP_trigger)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_trigger.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   if(!ipRecv.find("toggle")) return 0;
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      m_trigger = false;
      indi::updateSwitchIfChanged(m_indiP_trigger, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
   }
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      m_trigger = true;
      indi::updateSwitchIfChanged(m_indiP_trigger, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK);
   }

   return 0;
}

INDI_NEWCALLBACK_DEFN(dmModulator, m_indiP_modulating)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_modulating.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   if(!ipRecv.find("toggle")) return 0;
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      m_modulating = false;
      indi::updateSwitchIfChanged(m_indiP_modulating, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
   }
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      m_modulating = true;
      indi::updateSwitchIfChanged(m_indiP_modulating, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK);
   }
   
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(dmModulator, m_indiP_zero)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_zero.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }

   if(m_modulating == true)
   {
      log<text_log>("zero requested but currently modulating", logPrio::LOG_NOTICE);
      return 0;
   }

   if(!ipRecv.find("request")) return 0;
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      m_imageStream.md->write = 1;
   
      memset(m_imageStream.array.raw, 0, m_width*m_height*m_typeSize);
      timespec currtime;
      clock_gettime(CLOCK_REALTIME, &currtime);
      m_imageStream.md->atime = currtime;
      m_imageStream.md->writetime = currtime;
               
      m_imageStream.md->cnt0++;
   
      m_imageStream.md->write=0;
      ImageStreamIO_sempost(&m_imageStream,-1);      
      log<text_log>("zeroed");
   }
   
   
   return 0;
}


} //namespace app
} //namespace MagAOX

#endif //dmModulator_hpp
