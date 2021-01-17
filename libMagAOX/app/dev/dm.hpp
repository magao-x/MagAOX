/** \file dm.hpp
  * \brief The MagAO-X generic deformable mirror controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#ifndef dm_hpp
#define dm_hpp

/** Tests
 \todo test that restarting fpsCtrl doesn't scram this
 */

#include <mx/improc/eigenImage.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "../../ImageStreamIO/ImageStruct.hpp"

namespace MagAOX
{
namespace app
{
namespace dev 
{

template<typename typeT>
constexpr uint8_t ImageStreamTypeCode()
{
   return 0;
}

template<>
constexpr uint8_t ImageStreamTypeCode<float>()
{
   return _DATATYPE_FLOAT;
}

template<>
constexpr uint8_t ImageStreamTypeCode<double>()
{
   return _DATATYPE_DOUBLE; 
}

/** MagAO-X generic deformable mirror controller
  *
  * 
  * The derived class `derivedT` must expose the following interface
   \code 
    
   \endcode  
  * Each of the above functions should return 0 on success, and -1 on an error. 
  * 
  * This class should be declared a friend in the derived class, like so:
   \code 
    friend class dev::dm<derivedT,realT>;
   \endcode
  *
  * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic`, `appShutdown`, and `udpdateINDI`
  * functions must be placed in the derived class's functions of the same name.
  *
  * \ingroup appdev
  */
template<class derivedT, typename realT>
class dm 
{
   
protected:

   /** \name Configurable Parameters
    * @{
    */
   
   std::string m_calibPath; ///< The path to this DM's calibration files.
   std::string m_flatPath; ///< The path to this DM's flat files (usually the same as calibPath)
   std::string m_flat {"flat.fits"}; ///< The file name of the this DM's current flat.
   std::string m_test {"test.fits"}; ///< The file name of the this DM's current test command.
   
   std::string m_shmimFlat; ///< The name of the shmim stream to write the flat to.
   std::string m_shmimTest; ///< The name of the shmim stream to write the test to.
   std::string m_shmimSat; ///< The name of the shmim stream to write the saturation map to.
   std::string m_shmimSatPerc; ///< The name of the shmim stream to write the saturation percentage map to.
   
   int m_satAvgInt {100}; ///< The time in milliseconds to accumulate saturation over.
   
   int m_satThreadPrio {0}; ///< Priority of the saturation thread, should normally be > 0.
    
   uint32_t m_dmWidth {0}; ///< The width of the images in the stream
   uint32_t m_dmHeight {0}; ///< The height of the images in the stream
   
   static constexpr uint8_t m_dmDataType = ImageStreamTypeCode<realT>(); ///< The ImageStreamIO type code.
   
   ///@}
   
   
   std::string m_calibRelDir; ///< The directory relative to the calibPath.  Set this before calling dm<derivedT,realT>::loadConfig().
   
   mx::improc::eigenImage<realT> m_flatCommand; ///< Data storage for the flat command
   bool m_flatLoaded {false}; ///< Flag indicating whether a flat is loaded in memory
   
   IMAGE m_flatImageStream; ///< The ImageStreamIO shared memory buffer for the flat.
   bool m_flatSet {false}; ///< Flag indicating whether the flat command has been set.
   
   mx::improc::eigenImage<realT> m_testCommand; ///< Data storage for the test command
   bool m_testLoaded {false}; ///< Flag indicating whether a test command is loaded in memory
   
   IMAGE m_testImageStream; ///< The ImageStreamIO shared memory buffer for the test.
   bool m_testSet {false}; ///< Flag indicating whether the test command has been set.
   
public:

   /// Setup the configuration system
   /**
     * This should be called in `derivedT::setupConfig` as
     * \code
       dm<derivedT,realT>::setupConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void setupConfig(mx::app::appConfigurator & config /**< [out] the derived classes configurator*/);

   /// load the configuration system results
   /**
     * This should be called in `derivedT::loadConfig` as
     * \code
       dm<derivedT,realT>::loadConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void loadConfig(mx::app::appConfigurator & config /**< [in] the derived classes configurator*/);

   /// Startup function
   /** 
     * This should be called in `derivedT::appStartup` as
     * \code
       dm<derivedT,realT>::appStartup();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appStartup();

   /// DM application logic
   /** This should be called in `derivedT::appLogic` as
     * \code
       dm<derivedT,realT>::appLogic();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appLogic();

   /// DM shutdown
   /** This should be called in `derivedT::appShutdown` as
     * \code
       dm<derivedT,realT>::appShutdown();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appShutdown();
   
   /// Called after shmimMonitor connects to the dmXXdisp stream.  Checks for proper size.
   /**
     * \returns 0 on success
     * \returns -1 if incorrect size or data type in stream.
     */
   int allocate( const dev::shmimT & sp);
   
   /// Called by shmimMonitor when a new DM command is available.  This is just a pass-through to derivedT::commandDM(char*).
   int processImage( void * curr_src,
                     const dev::shmimT & sp
                   );
   
   int loadFlat(std::string & target);
   
   int loadTest(std::string & target);
   
   int setFlat();
   
   int zeroFlat();
   
   int setTest();
   
   int zeroTest();
   
protected:
   
   mx::improc::eigenImage<uint8_t> m_instSatMap; ///< The instantaneous saturation map, 0/1, set by the commandDM() function of the derived class.
   mx::improc::eigenImage<uint16_t> m_accumSatMap; ///< The accumulated saturation map, which acccumulates for m_satAvgInt then is publised as a 0/1 image. 
   mx::improc::eigenImage<float> m_satPercMap; ///< Map of the percentage of time each actator was saturated during the avg. interval.
   
   IMAGE m_satImageStream; ///< The ImageStreamIO shared memory buffer for the sat map.
   IMAGE m_satPercImageStream; ///< The ImageStreamIO shared memory buffer for the sat percentage map.
   
   /** \name Saturation Thread
     * This thread processes the saturation maps
     * @{
     */
   
   sem_t m_satSemaphore; ///< Semaphore used to tell the saturation thread to run.
   
   bool m_satThreadInit {true}; ///< Synchronizer for thread startup, to allow priority setting to finish.
   
   pid_t m_satThreadID {0}; ///< The ID of the saturation thread.
   
   pcf::IndiProperty m_satThreadProp; ///< The property to hold the saturation thread details.
   
   std::thread m_satThread; ///< A separate thread for the actual saturation processing

   ///Thread starter, called by MagAOXApp::threadStart on thread construction.  Calls satThreadExec.
   static void satThreadStart( dm * d /**< [in] a pointer to a dm instance (normally this) */);

   /// Execute saturation processing
   void satThreadExec();

   
   ///@}
   
protected:
   
    /** \name INDI 
      *
      *@{
      */ 
protected:
   //declare our properties
   
   pcf::IndiProperty m_indiP_flat; ///< Property used to set and report the current flat
   pcf::IndiProperty m_indiP_test; ///< Property used to set and report the current test
   
   pcf::IndiProperty m_indiP_init;
   pcf::IndiProperty m_indiP_zero;
   pcf::IndiProperty m_indiP_release;
   pcf::IndiProperty m_indiP_setFlat;
   pcf::IndiProperty m_indiP_zeroFlat;
   pcf::IndiProperty m_indiP_setTest;
   pcf::IndiProperty m_indiP_zeroTest;

public:

   /// The static callback function to be registered for changing the flat command file
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_flat( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                   const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                 );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_flat( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   
   /// The static callback function to be registered for changing the test command file
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_test( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                   const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                 );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_test( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   
   
   /// The static callback function to be registered for initializing the DM.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_init( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                   const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                 );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_init( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   /// The static callback function to be registered for initializing the DM.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_zero( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                   const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                 );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_zero( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   /// The static callback function to be registered for initializing the DM.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_release( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                   const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                 );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_release( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   /// The static callback function to be registered for setting the flat
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_setFlat( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                           const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                         );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_setFlat( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   
   
   /// The static callback function to be registered for zeroing the flat
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_zeroFlat( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                           const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                         );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_zeroFlat( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   
   
   /// The static callback function to be registered for setting the test shape
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_setTest( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                           const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                         );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_setTest( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// The static callback function to be registered for zeroing the test
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_zeroTest( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                           const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                         );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_zeroTest( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   
   
   
   /// Update the INDI properties for this device controller
   /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateINDI();

   ///@}
   
private:
   derivedT & derived()
   {
      return *static_cast<derivedT *>(this);
   }
};

template<class derivedT, typename realT>
void dm<derivedT,realT>::setupConfig(mx::app::appConfigurator & config)
{
   config.add("dm.calibPath", "", "dm.calibPath", argType::Required, "dm", "calibPath", false, "string", "The path to calibration files, relative to the MagAO-X calibration path.");
   
   config.add("dm.flatPath", "", "dm.flatPath", argType::Required, "dm", "flatPath", false, "string", "The path to flat files.  Default is the calibration path.");
     
   //Overriding the shmimMonitor setup so that these all go in the dm section
   //Otherwise, would call shmimMonitor<dm<derivedT,realT>>::setupConfig();
   config.add("dm.threadPrio", "", "dm.threadPrio", argType::Required, "dm", "threadPrio", false, "int", "The real-time priority of the dm control thread.");
   
   config.add("dm.shmimName", "", "dm.shmimName", argType::Required, "dm", "shmimName", false, "string", "The name of the ImageStreamIO shared memory image to monitor for DM comands. Will be used as /tmp/<shmimName>.im.shm.");
   
   config.add("dm.shmimFlat", "", "dm.shmimFlat", argType::Required, "dm", "shmimFlat", false, "string", "The name of the ImageStreamIO shared memory image to write the flat command to.  Default is shmimName with 00 apended (i.e. dm00disp -> dm00disp00). ");
   
   config.add("dm.shmimTest", "", "dm.shmimTest", argType::Required, "dm", "shmimTest", false, "string", "The name of the ImageStreamIO shared memory image to write the test command to.  Default is shmimName with 01 apended (i.e. dm00disp -> dm00disp01). ");
   
   config.add("dm.shmimSat", "", "dm.shmimSat", argType::Required, "dm", "shmimSat", false, "string", "The name of the ImageStreamIO shared memory image to write the saturation map to.  Default is shmimName with SA apended (i.e. dm00disp -> dm00dispSA).  This is created.");
   
   config.add("dm.shmimSatPerc", "", "dm.shmimSatPerc", argType::Required, "dm", "shmimSatPerc", false, "string", "The name of the ImageStreamIO shared memory image to write the saturation percentage map to.  Default is shmimName with SP apended (i.e. dm00disp -> dm00dispSP).  This is created.");
   
   config.add("dm.satAvgInt", "", "dm.satAvgInt", argType::Required, "dm", "satAvgInt", false, "int", "The interval in milliseconds over which saturation is accumulated before updating.  Default is 100 ms.");
   
   config.add("dm.width", "", "dm.width", argType::Required, "dm", "width", false, "string", "The width of the DM in actuators.");
   config.add("dm.height", "", "dm.height", argType::Required, "dm", "height", false, "string", "The height of the DM in actuators.");
}

template<class derivedT, typename realT>
void dm<derivedT,realT>::loadConfig(mx::app::appConfigurator & config)
{
   
   m_calibPath = derived().m_calibDir + "/" + m_calibRelDir;
   
   config( m_calibPath, "dm.calibPath");
   
   m_flatPath = m_calibPath;
   
   config( m_flatPath, "dm.flatPath");
  
   //Overriding the shmimMonitor setup so that these all go in the dm section
   //Otherwise, would call shmimMonitor<dm<derivedT,realT>>::loadConfig(config);
   config(derived().m_smThreadPrio, "dm.threadPrio");
   config(derived().m_shmimName, "dm.shmimName");
  
   m_shmimFlat = derived().m_shmimName + "00";
   config(m_shmimFlat, "dm.shmimFlat");
  
   m_shmimTest = derived().m_shmimName + "02";
   config(m_shmimTest, "dm.shmimTest");
   
   m_shmimSat = derived().m_shmimName + "ST";
   config(m_shmimSat, "dm.shmimSat");
   
   m_shmimSatPerc = derived().m_shmimName + "SP";
   config(m_shmimSatPerc, "dm.shmimSatPerc");
   
   config(m_satAvgInt, "dm.satAvgInt");
   
   config(m_dmWidth, "dm.width");
   config(m_dmHeight, "dm.height");
}
   

template<class derivedT, typename realT>
int dm<derivedT,realT>::appStartup()
{
   if( m_dmDataType == 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, "unsupported DM data type"});
      return -1;
   }
   
   //Register the flat INDI property
   m_indiP_flat = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_flat.setDevice(derived().configName());
   m_indiP_flat.setName("flat");
   m_indiP_flat.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_flat.setState(pcf::IndiProperty::Idle);
   m_indiP_flat.add(pcf::IndiElement("current"));
   m_indiP_flat["current"] = m_flat;
   m_indiP_flat.add(pcf::IndiElement("target"));
   m_indiP_flat["target"] = "";
   m_indiP_flat.add(pcf::IndiElement("path"));
   m_indiP_flat["path"] = m_flatPath;
   m_indiP_flat.add(pcf::IndiElement("shmimName"));
   m_indiP_flat["shmimName"] = m_shmimFlat;
   
   if( derived().registerIndiPropertyNew( m_indiP_flat, st_newCallBack_flat) < 0)
   {
      #ifndef DM_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   
   //Register the test INDI property
   m_indiP_test = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_test.setDevice(derived().configName());
   m_indiP_test.setName("test");
   m_indiP_test.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_test.setState(pcf::IndiProperty::Idle);
   m_indiP_test.add(pcf::IndiElement("current"));
   m_indiP_test["current"] = m_test;
   m_indiP_test.add(pcf::IndiElement("target"));
   m_indiP_test["target"] = "";
   m_indiP_test.add(pcf::IndiElement("shmimName"));
   m_indiP_test["shmimName"] = m_shmimTest;
   
   if( derived().registerIndiPropertyNew( m_indiP_test, st_newCallBack_test) < 0)
   {
      #ifndef DM_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   
   
   //Register the init INDI property
   m_indiP_init = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_init.setDevice(derived().configName());
   m_indiP_init.setName("initDM");
   m_indiP_init.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_init.setState(pcf::IndiProperty::Idle);
   m_indiP_init.add(pcf::IndiElement("request"));
   m_indiP_init["request"] = "";
   
   if( derived().registerIndiPropertyNew( m_indiP_init, st_newCallBack_init) < 0)
   {
      #ifndef DM_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the zero INDI property
   m_indiP_zero = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_zero.setDevice(derived().configName());
   m_indiP_zero.setName("zeroDM");
   m_indiP_zero.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_zero.setState(pcf::IndiProperty::Idle);
   m_indiP_zero.add(pcf::IndiElement("request"));
   m_indiP_zero["request"] = "";
   
   if( derived().registerIndiPropertyNew( m_indiP_zero, st_newCallBack_zero) < 0)
   {
      #ifndef DM_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the release INDI property
   m_indiP_release = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_release.setDevice(derived().configName());
   m_indiP_release.setName("releaseDM");
   m_indiP_release.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_release.setState(pcf::IndiProperty::Idle);
   m_indiP_release.add(pcf::IndiElement("request"));
   m_indiP_release["request"] = "";
   
   if( derived().registerIndiPropertyNew( m_indiP_release, st_newCallBack_release) < 0)
   {
      #ifndef DM_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the setFlat INDI property
   m_indiP_setFlat = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_setFlat.setDevice(derived().configName());
   m_indiP_setFlat.setName("setFlat");
   m_indiP_setFlat.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_setFlat.setState(pcf::IndiProperty::Idle);
   m_indiP_setFlat.add(pcf::IndiElement("request"));
   m_indiP_setFlat["request"] = "";
   
   if( derived().registerIndiPropertyNew( m_indiP_setFlat, st_newCallBack_setFlat) < 0)
   {
      #ifndef DM_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the zeroFlat INDI property
   m_indiP_zeroFlat = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_zeroFlat.setDevice(derived().configName());
   m_indiP_zeroFlat.setName("zeroFlat");
   m_indiP_zeroFlat.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_zeroFlat.setState(pcf::IndiProperty::Idle);
   m_indiP_zeroFlat.add(pcf::IndiElement("request"));
   m_indiP_zeroFlat["request"] = "";
   
   if( derived().registerIndiPropertyNew( m_indiP_zeroFlat, st_newCallBack_zeroFlat) < 0)
   {
      #ifndef DM_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   
   //Register the setTest INDI property
   m_indiP_setTest = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_setTest.setDevice(derived().configName());
   m_indiP_setTest.setName("setTest");
   m_indiP_setTest.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_setTest.setState(pcf::IndiProperty::Idle);
   m_indiP_setTest.add(pcf::IndiElement("request"));
   m_indiP_setTest["request"] = "";
   
   if( derived().registerIndiPropertyNew( m_indiP_setTest, st_newCallBack_setTest) < 0)
   {
      #ifndef DM_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the zeroTest INDI property
   m_indiP_zeroTest = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_zeroTest.setDevice(derived().configName());
   m_indiP_zeroTest.setName("zeroTest");
   m_indiP_zeroTest.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_zeroTest.setState(pcf::IndiProperty::Idle);
   m_indiP_zeroTest.add(pcf::IndiElement("request"));
   m_indiP_zeroTest["request"] = "";
   
   if( derived().registerIndiPropertyNew( m_indiP_zeroTest, st_newCallBack_zeroTest) < 0)
   {
      #ifndef DM_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   if(m_flat != "")
   {
      loadFlat(m_flat);
   }
   
   if(sem_init(&m_satSemaphore, 0,0) < 0) return derivedT::template log<software_critical, -1>({__FILE__, __LINE__, errno,0, "Initializing sat semaphore"});
   
   if(derived().threadStart( m_satThread, m_satThreadInit, m_satThreadID, m_satThreadProp, m_satThreadPrio, "saturation", this, satThreadStart) < 0)
   {
      derivedT::template log<software_error, -1>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;

}

template<class derivedT, typename realT>
int dm<derivedT,realT>::appLogic()
{
   //do a join check to see if other threads have exited.
   if(pthread_tryjoin_np(m_satThread.native_handle(),0) == 0)
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "saturation thread has exited"});
      
      return -1;
   }
   
   return 0;

}


template<class derivedT, typename realT>
int dm<derivedT,realT>::appShutdown()
{
   if(m_satThread.joinable())
   {
      pthread_kill(m_satThread.native_handle(), SIGUSR1);
      try
      {
         m_satThread.join(); //this will throw if it was already joined
      }
      catch(...)
      {
      }
   }
   
   return 0;
}



template<class derivedT, typename realT>
int dm<derivedT,realT>::allocate( const dev::shmimT & sp)
{
   static_cast<void>(sp); //be unused
   
   int err = 0;
   
   if(derived().m_width != m_dmWidth)
   {
      derivedT::template log<software_critical>({__FILE__,__LINE__, "shmim width does not match configured DM width"});
      ++err;
   }
   
   if(derived().m_height != m_dmHeight)
   {
      derivedT::template log<software_critical>({__FILE__,__LINE__, "shmim height does not match configured DM height"});
      ++err;
   }
   
   if(derived().m_dataType != m_dmDataType)
   {
      derivedT::template log<software_critical>({__FILE__,__LINE__, "shmim data type does not match configured DM data type"});
      ++err;
   }
   
   if(err) return -1;
   
   m_instSatMap.resize(m_dmWidth,m_dmHeight);
   m_instSatMap.setZero();
   
   m_accumSatMap.resize(m_dmWidth,m_dmHeight);
   m_accumSatMap.setZero();
   
   m_satPercMap.resize(m_dmWidth,m_dmHeight);
   m_satPercMap.setZero();
   
   return 0;
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::processImage( void * curr_src,
                                      const dev::shmimT & sp
                                    )
{
   static_cast<void>(sp); //be unused
   
   int rv = derived().commandDM( curr_src );
   
   if(rv < 0)
   {
      derivedT::template log<software_critical>({__FILE__, __LINE__, errno, rv, "Error from commandDM"});
      return rv;
   }
   //Tell the sat thread to get going
   if(sem_post(&m_satSemaphore) < 0)
   {
      derivedT::template log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
      return -1;
   }
   
   return rv;
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::loadFlat(std::string & target)
{
   
   ///\todo check path for /
   
   m_flat = target;
   std::string targetPath = m_flatPath + "/" + m_flat;
   
   //load into memory.
   mx::fits::fitsFile<realT> ff;
   if(ff.read(m_flatCommand, targetPath) < 0)
   {
      derivedT::template log<text_log>("flat file " + targetPath + " not found", logPrio::LOG_ERROR);
      return -1;
   }
   
   derivedT::template log<text_log>("loaded flat file " + targetPath);
   m_flatLoaded = true;
   
   indi::updateIfChanged(m_indiP_flat, "current", m_flat, derived().m_indiDriver);
   indi::updateIfChanged(m_indiP_flat, "target", m_flat, derived().m_indiDriver);
   
   if(m_flatSet) setFlat();
   
   return 0;
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::loadTest(std::string & target)
{
   
   ///\todo check path for /
   
   m_test = target;
   std::string targetPath = m_flatPath + "/" + m_test;
   
   //load into memory.
   mx::fits::fitsFile<realT> ff;
   if(ff.read(m_testCommand, targetPath) < 0)
   {
      derivedT::template log<text_log>("test file " + targetPath + " not found", logPrio::LOG_ERROR);
      return -1;
   }
   
   derivedT::template log<text_log>("loaded test file " + targetPath);
   m_testLoaded = true;
   
   indi::updateIfChanged(m_indiP_test, "current", m_test, derived().m_indiDriver);
   indi::updateIfChanged(m_indiP_test, "target", m_test, derived().m_indiDriver);
   
   if(m_testSet) setTest();
   
   return 0;
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::setFlat()
{
   if( ImageStreamIO_openIm(&m_flatImageStream, m_shmimFlat.c_str()) != 0)
   {
      derivedT::template log<text_log>("could not connect to flat channel " + m_shmimFlat, logPrio::LOG_WARNING);
      return -1;
   }
   
   if( m_flatImageStream.md[0].size[0] != m_dmWidth)
   {
      ImageStreamIO_closeIm(&m_flatImageStream);
      derivedT::template log<text_log>("width mismatch between " + m_shmimFlat + " and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   if( m_flatImageStream.md[0].size[1] != m_dmHeight)
   {
      ImageStreamIO_closeIm(&m_flatImageStream);
      derivedT::template log<text_log>("height mismatch between " + m_shmimFlat + " and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   if(!m_flatLoaded)
   {
      ImageStreamIO_closeIm(&m_flatImageStream);
      derivedT::template log<text_log>("no flat loaded", logPrio::LOG_ERROR);
      return -1;
   }

   if( m_flatCommand.rows() != m_dmWidth)
   {
      ImageStreamIO_closeIm(&m_flatImageStream);
      derivedT::template log<text_log>("width mismatch between flat file and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   if( m_flatCommand.cols() != m_dmHeight)
   {
      ImageStreamIO_closeIm(&m_flatImageStream);
      derivedT::template log<text_log>("height mismatch between flat file and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   m_flatImageStream.md->write=1;
   
   ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here anyway.  With bounds checks b/c not everyone handles cnt1 properly.
   //Copy
   memcpy( m_flatImageStream.array.raw, m_flatCommand.data(), m_dmWidth*m_dmHeight*sizeof(realT));
   
   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &m_flatImageStream.md->writetime);

   //Set the image acquisition timestamp
   m_flatImageStream.md->atime = m_flatImageStream.md->writetime;
         
   m_flatImageStream.md->cnt0++;
   m_flatImageStream.md->write=0;
   ImageStreamIO_sempost(&m_flatImageStream,-1);
         
   m_flatSet = true;
   
   //Post the semaphore
   ImageStreamIO_closeIm(&m_flatImageStream);
   
   derivedT::template log<text_log>("flat set");
   
   return 0;
}
  
template<class derivedT, typename realT>  
int dm<derivedT,realT>::zeroFlat()
{
   if( ImageStreamIO_openIm(&m_flatImageStream, m_shmimFlat.c_str()) != 0)
   {
      derivedT::template log<text_log>("could not connect to flat channel " + m_shmimFlat, logPrio::LOG_WARNING);
      return -1;
   }
   
   if( m_flatImageStream.md[0].size[0] != m_dmWidth)
   {
      ImageStreamIO_closeIm(&m_flatImageStream);
      derivedT::template log<text_log>("width mismatch between " + m_shmimFlat + " and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   if( m_flatImageStream.md[0].size[1] != m_dmHeight)
   {
      ImageStreamIO_closeIm(&m_flatImageStream);
      derivedT::template log<text_log>("height mismatch between " + m_shmimFlat + " and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   m_flatImageStream.md->write=1;
   
   ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here anyway.  With bounds checks b/c not everyone handles cnt1 properly.
   //Zero
   memset( m_flatImageStream.array.raw, 0, m_dmWidth*m_dmHeight*sizeof(realT));
   
   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &m_flatImageStream.md->writetime);

   //Set the image acquisition timestamp
   m_flatImageStream.md->atime = m_flatImageStream.md->writetime;
         
   m_flatImageStream.md->cnt0++;
   m_flatImageStream.md->write=0;
   ImageStreamIO_sempost(&m_flatImageStream,-1);
         
   m_flatSet = false;
   
   //Post the semaphore
   ImageStreamIO_closeIm(&m_flatImageStream);
   
   derivedT::template log<text_log>("flat zeroed");
   
   return 0;
}
  
template<class derivedT, typename realT>  
int dm<derivedT,realT>::setTest()
{
   if( ImageStreamIO_openIm(&m_testImageStream, m_shmimTest.c_str()) != 0)
   {
      derivedT::template log<text_log>("could not connect to test channel " + m_shmimTest, logPrio::LOG_WARNING);
      return -1;
   }
   
   if( m_testImageStream.md->size[0] != m_dmWidth)
   {
      ImageStreamIO_closeIm(&m_testImageStream);
      derivedT::template log<text_log>("width mismatch between " + m_shmimTest + " and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   if( m_testImageStream.md->size[1] != m_dmHeight)
   {
      ImageStreamIO_closeIm(&m_testImageStream);
      derivedT::template log<text_log>("height mismatch between " + m_shmimTest + " and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   if(!m_testLoaded)
   {
      ImageStreamIO_closeIm(&m_testImageStream);
      derivedT::template log<text_log>("no test loaded", logPrio::LOG_ERROR);
      return -1;
   }

   if( m_testCommand.rows() != m_dmWidth)
   {
      ImageStreamIO_closeIm(&m_testImageStream);
      derivedT::template log<text_log>("width mismatch between test file and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   if( m_testCommand.cols() != m_dmHeight)
   {
      ImageStreamIO_closeIm(&m_testImageStream);
      derivedT::template log<text_log>("height mismatch between test file and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   m_testImageStream.md->write=1;
   
   ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here anyway.  With bounds checks b/c not everyone handles cnt1 properly.
   //Copy
   memcpy( m_testImageStream.array.raw, m_testCommand.data(), m_dmWidth*m_dmHeight*sizeof(realT));
   
   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &m_testImageStream.md->writetime);

   //Set the image acquisition timestamp
   m_testImageStream.md->atime = m_testImageStream.md->writetime;
         
   m_testImageStream.md->cnt0++;
   m_testImageStream.md->write=0;
   ImageStreamIO_sempost(&m_testImageStream,-1);
         
   m_testSet = true;
   
   //Post the semaphore
   ImageStreamIO_closeIm(&m_testImageStream);
   
   derivedT::template log<text_log>("test set");
   
   return 0;
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::zeroTest()
{
   if( ImageStreamIO_openIm(&m_testImageStream, m_shmimTest.c_str()) != 0)
   {
      derivedT::template log<text_log>("could not connect to test channel " + m_shmimTest, logPrio::LOG_WARNING);
      return -1;
   }
   
   if( m_testImageStream.md[0].size[0] != m_dmWidth)
   {
      ImageStreamIO_closeIm(&m_testImageStream);
      derivedT::template log<text_log>("width mismatch between " + m_shmimTest + " and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   if( m_testImageStream.md[0].size[1] != m_dmHeight)
   {
      ImageStreamIO_closeIm(&m_testImageStream);
      derivedT::template log<text_log>("height mismatch between " + m_shmimTest + " and configured DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   m_testImageStream.md->write=1;
   
   ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here anyway.  With bounds checks b/c not everyone handles cnt1 properly.
   //Zero
   memset( m_testImageStream.array.raw, 0, m_dmWidth*m_dmHeight*sizeof(realT));
   
   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &m_testImageStream.md->writetime);

   //Set the image acquisition timestamp
   m_testImageStream.md->atime = m_testImageStream.md->writetime;
         
   m_testImageStream.md->cnt0++;
   m_testImageStream.md->write=0;
   ImageStreamIO_sempost(&m_testImageStream,-1);
         
   m_testSet = false;
   
   //Post the semaphore
   ImageStreamIO_closeIm(&m_testImageStream);
   
   derivedT::template log<text_log>("test zeroed");
   
   return 0;
}

template<class derivedT, typename realT>
void dm<derivedT,realT>::satThreadStart(dm *d)
{
   d->satThreadExec();
}

template<class derivedT, typename realT>
void dm<derivedT,realT>::satThreadExec()
{
   //Get the thread PID immediately so the caller can return.
   m_satThreadID = syscall(SYS_gettid);
   
   //Wait for the thread starter to finish initializing this thread.
   while(m_satThreadInit == true && derived().shutdown() == 0)
   {
      sleep(1);
   }
   if(derived().shutdown()) return;
   
   uint32_t imsize[3] = {0,0,0};
   
   //Check for allocation to have happened.
   while((m_accumSatMap.rows() == 0 || m_accumSatMap.cols() == 0) && !derived().shutdown())
   {
      sleep(1);
   }
   if(derived().shutdown()) return;

   imsize[0] = m_dmWidth; 
   imsize[1] = m_dmHeight;
   imsize[2] = 1;
      
   std::cerr << "Creating: " << m_shmimSat << " " << m_dmWidth << " " << m_dmHeight << " " << 1 << "\n";
      
   ImageStreamIO_createIm_gpu(&m_satImageStream, m_shmimSat.c_str(), 3, imsize, IMAGESTRUCT_UINT8, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL);
   ImageStreamIO_createIm_gpu(&m_satPercImageStream, m_shmimSatPerc.c_str(), 3, imsize, IMAGESTRUCT_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL);
       
   bool opened = true;
   
   m_satImageStream.md->cnt1 = 0;
   m_satPercImageStream.md->cnt1 = 0;
   
   //This is the working memory for making the 1/0 mask out of m_accumSatMap
   mx::improc::eigenImage<uint8_t> satmap(m_dmWidth, m_dmHeight);
   
   int naccum = 0;
   double t_accumst = mx::sys::get_curr_time();
   
   //This is the main image grabbing loop.      
   while(!derived().shutdown())
   {
      //Get timespec for sem_timedwait
      timespec ts;
      if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
      {
         derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); 
         return;
      }
      ts.tv_sec += 1;
      
      //Wait on semaphore
      if(sem_timedwait(&m_satSemaphore, &ts) == 0)
      {
         //not a timeout -->accumulate
         for(int rr=0; rr < m_instSatMap.rows(); ++rr)
         {
            for(int cc=0; cc< m_instSatMap.cols(); ++cc)
            {
               m_accumSatMap(rr,cc) += m_instSatMap(rr,cc);
            }
         }
         ++naccum;

         // If less than avg int --> go back and wait again
         if(mx::sys::get_curr_time(ts) - t_accumst < m_satAvgInt/1000.0) continue;
         
         // If greater than avg int --> calc stats, write to streams.
         for(int rr=0; rr < m_instSatMap.rows(); ++rr)
         {
            for(int cc=0; cc< m_instSatMap.cols(); ++cc)
            {
               m_satPercMap(rr,cc) = m_accumSatMap(rr,cc)/naccum;           
               satmap(rr,cc) = (m_accumSatMap(rr,cc) > 0); //it's  1/0 map
            }
         }
      
         m_satImageStream.md->write=1;
         m_satPercImageStream.md->write=1;
         
         memcpy( m_satImageStream.array.raw, satmap.data() , m_dmWidth*m_dmHeight*sizeof(uint8_t));
         memcpy( m_satPercImageStream.array.raw, m_satPercMap.data() , m_dmWidth*m_dmHeight*sizeof(float));
         
         //Set the time of last write
         clock_gettime(CLOCK_REALTIME, &m_satImageStream.md->writetime);
         m_satPercImageStream.md->writetime = m_satImageStream.md->writetime;

         //Set the image acquisition timestamp
         m_satImageStream.md->atime = m_satImageStream.md->writetime;
         m_satPercImageStream.md->atime = m_satPercImageStream.md->writetime;
         
         //Update cnt1
         m_satImageStream.md->cnt1 = 0;
         m_satPercImageStream.md->cnt1 = 0;
          
         //Update cnt0
         m_satImageStream.md->cnt0++;
         m_satPercImageStream.md->cnt0++;
         
         m_satImageStream.writetimearray[0] = m_satImageStream.md->writetime;
         m_satImageStream.atimearray[0] = m_satImageStream.md->atime;
         m_satImageStream.cntarray[0] = m_satImageStream.md->cnt0;
         
         m_satPercImageStream.writetimearray[0] = m_satPercImageStream.md->writetime;
         m_satPercImageStream.atimearray[0] = m_satPercImageStream.md->atime;
         m_satPercImageStream.cntarray[0] = m_satPercImageStream.md->cnt0;
         
         //And post
         m_satImageStream.md->write=0;
         ImageStreamIO_sempost(&m_satImageStream,-1);
         
         m_satPercImageStream.md->write=0;
         ImageStreamIO_sempost(&m_satPercImageStream,-1);
         
         m_accumSatMap.setZero();
         naccum = 0;
         t_accumst = mx::sys::get_curr_time(ts);
      }
      else
      {
         //Check for why we timed out
         if(errno == EINTR) break; //This indicates signal interrupted us, time to restart or shutdown, loop will exit normally if flags set.
            
         //ETIMEDOUT just means we should wait more.
         //Otherwise, report an error.
         if(errno != ETIMEDOUT)
         {
            derivedT::template log<software_error>({__FILE__, __LINE__,errno, "sem_timedwait"});
            break;
         }
      }
   }

   if(opened)
   {
      ImageStreamIO_destroyIm( &m_satImageStream );
  
      ImageStreamIO_destroyIm( &m_satPercImageStream );
   }
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::updateINDI()
{
   if( !derived().m_indiDriver ) return 0;
   
   if(m_flatSet)
   {
      derived().updateIfChanged(m_indiP_flat, "current", m_flat);
      derived().updateIfChanged(m_indiP_flat, "target", m_flat);
   }
   else
   {
      derived().updateIfChanged(m_indiP_flat, "current", m_flat);
      derived().updateIfChanged(m_indiP_flat, "target", std::string(""));
   }
   
   
   if(m_testSet)
   {
      derived().updateIfChanged(m_indiP_test, "current", m_test);
      derived().updateIfChanged(m_indiP_test, "target", m_test);
   }
   else
   {
      derived().updateIfChanged(m_indiP_test, "current", m_test);
      derived().updateIfChanged(m_indiP_test, "target", std::string(""));
   }
   
   return 0;
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::st_newCallBack_flat( void * app,
                                             const pcf::IndiProperty &ipRecv
                                           )
{
   return static_cast<derivedT *>(app)->newCallBack_flat(ipRecv);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::newCallBack_flat( const pcf::IndiProperty &ipRecv )
{
   if ( ipRecv.getName() != m_indiP_flat.getName())
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
   }
   
   std::string target, current;
   
   if(ipRecv.find("target"))
   {
      target = ipRecv["target"].get<std::string>();
   }
   
   if(target == "" )
   {
      if(ipRecv.find("current"))
      {
         target = ipRecv["current"].get<std::string>();
      }
   }
   
   if(target == "") return 0;
   
   return loadFlat(target);
}


template<class derivedT, typename realT>
int dm<derivedT,realT>::st_newCallBack_test( void * app,
                                       const pcf::IndiProperty &ipRecv
                                     )
{
   return static_cast<derivedT *>(app)->newCallBack_test(ipRecv);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::newCallBack_test( const pcf::IndiProperty &ipRecv )
{
   if ( ipRecv.getName() != m_indiP_test.getName())
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
   }
   
   std::string target, current;
   
   if(ipRecv.find("target"))
   {
      target = ipRecv["target"].get<std::string>();
   }
   
   if(target == "" )
   {
      if(ipRecv.find("current"))
      {
         target = ipRecv["current"].get<std::string>();
      }
   }
   
   if(target == "") return 0;
   
   return loadTest(target);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::st_newCallBack_init( void * app,
                                       const pcf::IndiProperty &ipRecv
                                     )
{
   return static_cast<derivedT *>(app)->newCallBack_init(ipRecv);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::newCallBack_init( const pcf::IndiProperty &ipRecv )
{
   if ( ipRecv.getName() != m_indiP_init.getName())
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
   }
   
   if(!ipRecv.find("request")) return 0;
   
   std::string request = ipRecv["request"].get();
      
   if(request == "") return 0;
   
   return derived().initDM();
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::st_newCallBack_zero( void * app,
                                       const pcf::IndiProperty &ipRecv
                                     )
{
   return static_cast<derivedT *>(app)->newCallBack_zero(ipRecv);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::newCallBack_zero( const pcf::IndiProperty &ipRecv )
{
   if ( ipRecv.getName() != m_indiP_zero.getName())
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
   }
   
   if(!ipRecv.find("request")) return 0;
   
   std::string request = ipRecv["request"].get();
      
   if(request == "") return 0;
   
   return derived().zeroDM();
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::st_newCallBack_release( void * app,
                                       const pcf::IndiProperty &ipRecv
                                     )
{
   return static_cast<derivedT *>(app)->newCallBack_release(ipRecv);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::newCallBack_release( const pcf::IndiProperty &ipRecv )
{
   if ( ipRecv.getName() != m_indiP_release.getName())
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
   }
   
   if(!ipRecv.find("request")) return 0;
   
   std::string request = ipRecv["request"].get();
      
   if(request == "") return 0;
   
   return derived().releaseDM();
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::st_newCallBack_setFlat( void * app,
                                           const pcf::IndiProperty &ipRecv
                                         )
{
   return static_cast<derivedT *>(app)->newCallBack_setFlat(ipRecv);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::newCallBack_setFlat( const pcf::IndiProperty &ipRecv )
{
   if ( ipRecv.getName() != m_indiP_setFlat.getName())
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
   }
   
   if(!ipRecv.find("request")) return 0;
   
   std::string request = ipRecv["request"].get();
      
   if(request == "") return 0;
   
   return setFlat();
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::st_newCallBack_zeroFlat( void * app,
                                           const pcf::IndiProperty &ipRecv
                                         )
{
   return static_cast< derivedT *>(app)->newCallBack_zeroFlat(ipRecv);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::newCallBack_zeroFlat( const pcf::IndiProperty &ipRecv )
{
   if ( ipRecv.getName() != m_indiP_zeroFlat.getName())
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
   }
   
   if(!ipRecv.find("request")) return 0;
   
   std::string request = ipRecv["request"].get();
      
   if(request == "") return 0;
   
   return zeroFlat();
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::st_newCallBack_setTest( void * app,
                                           const pcf::IndiProperty &ipRecv
                                         )
{
   return static_cast< derivedT *>(app)->newCallBack_setTest(ipRecv);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::newCallBack_setTest( const pcf::IndiProperty &ipRecv )
{
   if ( ipRecv.getName() != m_indiP_setTest.getName())
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
   }
   
   if(!ipRecv.find("request")) return 0;
   
   std::string request = ipRecv["request"].get();
      
   if(request == "") return 0;
   
   return setTest();
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::st_newCallBack_zeroTest( void * app,
                                           const pcf::IndiProperty &ipRecv
                                         )
{
   return static_cast< derivedT *>(app)->newCallBack_zeroTest(ipRecv);
}

template<class derivedT, typename realT>
int dm<derivedT,realT>::newCallBack_zeroTest( const pcf::IndiProperty &ipRecv )
{
   if ( ipRecv.getName() != m_indiP_zeroTest.getName())
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
   }
   
   if(!ipRecv.find("request")) return 0;
   
   std::string request = ipRecv["request"].get();
      
   if(request == "") return 0;
   
   return zeroTest();
}

} //namespace dev
} //namespace app
} //namespace MagAOX
#endif
