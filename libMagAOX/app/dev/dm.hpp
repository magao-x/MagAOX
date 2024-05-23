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

#include <boost/filesystem/operations.hpp>

#include "../../ImageStreamIO/ImageStruct.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

template <typename typeT>
constexpr uint8_t ImageStreamTypeCode()
{
    return 0;
}

template <>
constexpr uint8_t ImageStreamTypeCode<float>()
{
    return _DATATYPE_FLOAT;
}

template <>
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
template <class derivedT, typename realT>
class dm
{

protected:
    /** \name Configurable Parameters
     * @{
     */

    std::string m_calibPath; ///< The path to this DM's calibration files.
    std::string m_flatPath;  ///< The path to this DM's flat files (usually the same as calibPath)
    std::string m_testPath;  ///< The path to this DM's test files (default is calibPath/tests;

    std::string m_flatDefault; ///< The file name of the this DM's default flat command. Path and extension will be ignored and can be omitted.
    std::string m_testDefault; ///< The file name of the this DM's default test command. Path and extension will be ignored and can be omitted.

    std::string m_shmimFlat;    ///< The name of the shmim stream to write the flat to.
    std::string m_shmimTest;    ///< The name of the shmim stream to write the test to.
    std::string m_shmimSat;     ///< The name of the shmim stream to write the saturation map to.
    std::string m_shmimSatPerc; ///< The name of the shmim stream to write the saturation percentage map to.

    int m_satAvgInt{100}; ///< The time in milliseconds to accumulate saturation over.

    ///\todo satThreadPrio configuration is not actually implemented.
    int m_satThreadPrio{0}; ///< Priority of the saturation thread, should normally be > 0.

    uint32_t m_dmWidth{0};  ///< The width of the images in the stream
    uint32_t m_dmHeight{0}; ///< The height of the images in the stream

    static constexpr uint8_t m_dmDataType = ImageStreamTypeCode<realT>(); ///< The ImageStreamIO type code.

    float m_percThreshold{0.98};         // percentage of frames saturated over interval
    float m_intervalSatThreshold{0.50};  //
    int m_intervalSatCountThreshold{10}; //

    std::vector<std::string> m_satTriggerDevice;
    std::vector<std::string> m_satTriggerProperty;

    ///@}

    std::string m_calibRelDir; ///< The directory relative to the calibPath.  Set this before calling dm<derivedT,realT>::loadConfig().

    int m_channels{0}; ///< The number of dmcomb channels found as part of allocation.

    std::map<std::string, std::string> m_flatCommands; ///< Map of flat file name to full path
    std::string m_flatCurrent;                         ///< The name of the current flat command

    mx::improc::eigenImage<realT> m_flatCommand; ///< Data storage for the flat command
    bool m_flatLoaded{false};                    ///< Flag indicating whether a flat is loaded in memory

    IMAGE m_flatImageStream; ///< The ImageStreamIO shared memory buffer for the flat.
    bool m_flatSet{false};   ///< Flag indicating whether the flat command has been set.

    std::map<std::string, std::string> m_testCommands; ///< Map of test file name to full path
    std::string m_testCurrent;

    mx::improc::eigenImage<realT> m_testCommand; ///< Data storage for the test command
    bool m_testLoaded{false};                    ///< Flag indicating whether a test command is loaded in memory

    IMAGE m_testImageStream; ///< The ImageStreamIO shared memory buffer for the test.
    bool m_testSet{false};   ///< Flag indicating whether the test command has been set.

    int m_overSatAct{0};         // counter
    int m_intervalSatExceeds{0}; // counter
    bool m_intervalSatTrip{0};   // flag to trip the loop opening

public:
    /// Setup the configuration system
    /**
      * This should be called in `derivedT::setupConfig` as
      * \code
        dm<derivedT,realT>::setupConfig(config);
        \endcode
      * with appropriate error checking.
      */
    int setupConfig(mx::app::appConfigurator &config /**< [out] the derived classes configurator*/);

    /// load the configuration system results
    /**
      * This should be called in `derivedT::loadConfig` as
      * \code
        dm<derivedT,realT>::loadConfig(config);
        \endcode
      * with appropriate error checking.
      */
    int loadConfig(mx::app::appConfigurator &config /**< [in] the derived classes configurator*/);

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

    /// DM Poweroff
    /** This should be called in `derivedT::onPowerOff` as
      * \code
        dm<derivedT,realT>::onPowerOff();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int onPowerOff();

    /// DM Poweroff Updates
    /** This should be called in `derivedT::whilePowerOff` as
      * \code
        dm<derivedT,realT>::whilePowerOff();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int whilePowerOff();

    /// Find the DM comb channels
    /** Introspectively finds all dmXXdispYY channels, zeroes them, and raises the semapahore
     * on the last to cause dmcomb to update.
     */
    int findDMChannels();

    /// Called after shmimMonitor connects to the dmXXdisp stream.  Checks for proper size.
    /**
     * \returns 0 on success
     * \returns -1 if incorrect size or data type in stream.
     */
    int allocate(const dev::shmimT &sp);

    /// Called by shmimMonitor when a new DM command is available.  This is just a pass-through to derivedT::commandDM(char*).
    int processImage(void *curr_src,
                     const dev::shmimT &sp);

    /// Calls derived()->releaseDM() and then 0s all channels and the sat map.
    /** This is called by the relevant INDI callback
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int releaseDM();

    /// Check the flats directory and update the list of flats if anything changes
    /** This is called once per appLogic and whilePowerOff loops.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int checkFlats();

    /// Load a flat file
    /** Uses the target argument for lookup in m_flatCommands to find the path
     * and loads the command in the local memory.  Calls setFlat if the flat
     * is currently set.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadFlat(const std::string &target /**< [in] the name of the flat to load */);

    /// Send the current flat command to the DM
    /** Writes the command to the designated shmim.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int setFlat(bool update = false /**< [in] If true, this is an update rather than a new set*/);

    /// Zero the flat command on the DM
    /** Writes a 0 array the designated shmim.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int zeroFlat();

    /// Check the tests directory and update the list of tests if anything changes
    /** This is called once per appLogic and whilePowerOff loops.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int checkTests();

    /// Load a test file
    /** Uses the target argument for lookup in m_testCommands to find the path
     * and loads the command in the local memory.  Calls setTest if the test
     * is currently set.
     */
    int loadTest(const std::string &target);

    /// Send the current test command to the DM
    /** Writes the command to the designated shmim.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int setTest();

    /// Zero the test command on the DM
    /** Writes a 0 array the designated shmim.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int zeroTest();

    /// Zero all channels
    /**
     * \returns 0 on sucess
     * \returns \<0 on an error
     */
    int zeroAll(bool nosem = false /**< [in] [optional] if true then the semaphore is not raised after zeroing all channels*/);

protected:
    mx::improc::eigenImage<uint8_t> m_instSatMap;   ///< The instantaneous saturation map, 0/1, set by the commandDM() function of the derived class.
    mx::improc::eigenImage<uint16_t> m_accumSatMap; ///< The accumulated saturation map, which acccumulates for m_satAvgInt then is publised as a 0/1 image.
    mx::improc::eigenImage<float> m_satPercMap;     ///< Map of the percentage of time each actator was saturated during the avg. interval.

    IMAGE m_satImageStream;     ///< The ImageStreamIO shared memory buffer for the sat map.
    IMAGE m_satPercImageStream; ///< The ImageStreamIO shared memory buffer for the sat percentage map.

    /// Clear the saturation maps and zero the shared membory.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int clearSat();

    /** \name Saturation Thread
     * This thread processes the saturation maps
     * @{
     */

    sem_t m_satSemaphore; ///< Semaphore used to tell the saturation thread to run.

    bool m_satThreadInit{true}; ///< Synchronizer for thread startup, to allow priority setting to finish.

    pid_t m_satThreadID{0}; ///< The ID of the saturation thread.

    pcf::IndiProperty m_satThreadProp; ///< The property to hold the saturation thread details.

    std::thread m_satThread; ///< A separate thread for the actual saturation processing

    /// Thread starter, called by MagAOXApp::threadStart on thread construction.  Calls satThreadExec.
    static void satThreadStart(dm *d /**< [in] a pointer to a dm instance (normally this) */);

    /// Execute saturation processing
    void satThreadExec();

    void intervalSatTrip()
    {
        if (m_satTriggerDevice.size() > 0 && m_satTriggerProperty.size() == m_satTriggerDevice.size())
        {
            for (size_t n = 0; n < m_satTriggerDevice.size(); ++n)
            {
                // We just silently fail
                try
                {
                    pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);

                    ipFreq.setDevice(m_satTriggerDevice[n]);
                    ipFreq.setName(m_satTriggerProperty[n]);
                    ipFreq.add(pcf::IndiElement("toggle"));
                    ipFreq["toggle"] = pcf::IndiElement::Off;
                    derived().sendNewProperty(ipFreq);

                    derivedT::template log<text_log>("DM saturation threshold exceeded.  Loop opened.", logPrio::LOG_WARNING);
                }
                catch (...)
                {
                }
            }
        }
    }

    ///@}

protected:
    /** \name INDI
     *
     *@{
     */
protected:
    // declare our properties

    pcf::IndiProperty m_indiP_flat; ///< Property used to set and report the current flat

    pcf::IndiProperty m_indiP_init;
    pcf::IndiProperty m_indiP_zero;
    pcf::IndiProperty m_indiP_release;

    pcf::IndiProperty m_indiP_flats;     ///< INDI Selection switch containing the flat files.
    pcf::IndiProperty m_indiP_flatShmim; ///< Publish the shmim being used for the flat
    pcf::IndiProperty m_indiP_setFlat;   ///< INDI toggle switch to set the current flat.

    pcf::IndiProperty m_indiP_tests;     ///< INDI Selection switch containing the test pattern files.
    pcf::IndiProperty m_indiP_testShmim; ///< Publish the shmim being used for the test command
    pcf::IndiProperty m_indiP_setTest;   ///< INDI toggle switch to set the current test pattern.

    pcf::IndiProperty m_indiP_zeroAll;

public:
    /// The static callback function to be registered for initializing the DM.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_init(void *app,                      ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                   const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_init(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

    /// The static callback function to be registered for initializing the DM.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_zero(void *app,                      ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                   const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_zero(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

    /// The static callback function to be registered for initializing the DM.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_release(void *app,                      ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                      const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_release(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

    /// The static callback function to be registered for selecting the flat file
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_flats(void *app,                      ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                    const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_flats(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

    /// The static callback function to be registered for setting the flat
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_setFlat(void *app,                      ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                      const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_setFlat(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

    /// The static callback function to be registered for selecting the test file
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_tests(void *app,                      ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                    const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_tests(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

    /// The static callback function to be registered for setting the test shape
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_setTest(void *app,                      ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                      const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_setTest(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

    /// The static callback function to be registered for zeroing all channels
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_zeroAll(void *app,                      ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                      const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
    );

    /// The callback for the zeroAll toggle switch, called by the static version
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_zeroAll(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

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
    derivedT &derived()
    {
        return *static_cast<derivedT *>(this);
    }
};

template <class derivedT, typename realT>
int dm<derivedT, realT>::setupConfig(mx::app::appConfigurator &config)
{
    config.add("dm.calibPath", "", "dm.calibPath", argType::Required, "dm", "calibPath", false, "string", "The path to calibration files, relative to the MagAO-X calibration path.");

    config.add("dm.flatPath", "", "dm.flatPath", argType::Required, "dm", "flatPath", false, "string", "The path to flat files.  Default is the calibration path.");
    config.add("dm.flatDefault", "", "dm.flatDefault", argType::Required, "dm", "flatDefault", false, "string", "The default flat file (path and extension are not required).");

    config.add("dm.testPath", "", "dm.testPath", argType::Required, "dm", "testPath", false, "string", "The path to test files.  Default is the calibration path plus /tests.");
    config.add("dm.testDefault", "", "dm.testDefault", argType::Required, "dm", "testDefault", false, "string", "The default test file (path and extension are not required).");

    // Overriding the shmimMonitor setup so that these all go in the dm section
    // Otherwise, would call shmimMonitor<dm<derivedT,realT>>::setupConfig();
    ///\todo we shmimMonitor now has configSection so this isn't necessary.
    config.add("dm.threadPrio", "", "dm.threadPrio", argType::Required, "dm", "threadPrio", false, "int", "The real-time priority of the dm control thread.");
    config.add("dm.cpuset", "", "dm.cpuset", argType::Required, "dm", "cpuset", false, "int", "The cpuset for the dm control thread.");

    config.add("dm.shmimName", "", "dm.shmimName", argType::Required, "dm", "shmimName", false, "string", "The name of the ImageStreamIO shared memory image to monitor for DM comands. Will be used as /tmp/<shmimName>.im.shm.");

    config.add("dm.shmimFlat", "", "dm.shmimFlat", argType::Required, "dm", "shmimFlat", false, "string", "The name of the ImageStreamIO shared memory image to write the flat command to.  Default is shmimName with 00 apended (i.e. dm00disp -> dm00disp00). ");

    config.add("dm.shmimTest", "", "dm.shmimTest", argType::Required, "dm", "shmimTest", false, "string", "The name of the ImageStreamIO shared memory image to write the test command to.  Default is shmimName with 01 apended (i.e. dm00disp -> dm00disp01). ");

    config.add("dm.shmimSat", "", "dm.shmimSat", argType::Required, "dm", "shmimSat", false, "string", "The name of the ImageStreamIO shared memory image to write the saturation map to.  Default is shmimName with SA apended (i.e. dm00disp -> dm00dispSA).  This is created.");

    config.add("dm.shmimSatPerc", "", "dm.shmimSatPerc", argType::Required, "dm", "shmimSatPerc", false, "string", "The name of the ImageStreamIO shared memory image to write the saturation percentage map to.  Default is shmimName with SP apended (i.e. dm00disp -> dm00dispSP).  This is created.");

    config.add("dm.satAvgInt", "", "dm.satAvgInt", argType::Required, "dm", "satAvgInt", false, "int", "The interval in milliseconds over which saturation is accumulated before updating.  Default is 100 ms.");

    config.add("dm.width", "", "dm.width", argType::Required, "dm", "width", false, "string", "The width of the DM in actuators.");
    config.add("dm.height", "", "dm.height", argType::Required, "dm", "height", false, "string", "The height of the DM in actuators.");

    config.add("dm.percThreshold", "", "dm.percThreshold", argType::Required, "dm", "percThreshold", false, "float", "Threshold on percentage of frames an actuator is saturated over an interval.  Default is 0.98.");
    config.add("dm.intervalSatThreshold", "", "dm.intervalSatThreshold", argType::Required, "dm", "intervalSatThreshold", false, "float", "Threshold on percentage of actuators which exceed percThreshold in an interval.  Default is 0.5.");
    config.add("dm.intervalSatCountThreshold", "", "dm.intervalSatCountThreshold", argType::Required, "dm", "intervalSatCountThreshold", false, "float", "Threshold one number of consecutive intervals the intervalSatThreshold is exceeded.  Default is 10.");

    config.add("dm.satTriggerDevice", "", "dm.satTriggerDevice", argType::Required, "dm", "satTriggerDevice", false, "vector<string>", "Device(s) with a toggle switch to toggle on saturation trigger.");
    config.add("dm.satTriggerProperty", "", "dm.satTriggerProperty", argType::Required, "dm", "satTriggerProperty", false, "vector<string>", "Property with a toggle switch to toggle on saturation trigger, one per entry in satTriggerDevice.");

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::loadConfig(mx::app::appConfigurator &config)
{

    m_calibPath = derived().m_calibDir + "/" + m_calibRelDir;
    config(m_calibPath, "dm.calibPath");

    // setup flats
    m_flatPath = m_calibPath + "/flats";
    config(m_flatPath, "dm.flatPath");

    config(m_flatDefault, "dm.flatDefault");
    if (m_flatDefault != "")
    {
        m_flatDefault = mx::ioutils::pathStem(m_flatDefault); // strip off path and extension if provided.
        m_flatCurrent = "default";
    }

    // setup tests
    m_testPath = m_calibPath + "/tests";
    config(m_testPath, "dm.testPath");

    config(m_testDefault, "dm.testDefault");
    if (m_testDefault != "")
    {
        m_testDefault = mx::ioutils::pathStem(m_testDefault); // strip off path and extension if provided.
        m_testCurrent = "default";
    }

    // Overriding the shmimMonitor setup so that these all go in the dm section
    // Otherwise, would call shmimMonitor<dm<derivedT,realT>>::loadConfig(config);
    config(derived().m_smThreadPrio, "dm.threadPrio");
    config(derived().m_smCpuset, "dm.cpuset");

    config(derived().m_shmimName, "dm.shmimName");

    if (derived().m_shmimName != "")
    {
        m_shmimFlat = derived().m_shmimName + "00";
        config(m_shmimFlat, "dm.shmimFlat");

        m_shmimTest = derived().m_shmimName + "02";
        config(m_shmimTest, "dm.shmimTest");

        m_shmimSat = derived().m_shmimName + "ST";
        config(m_shmimSat, "dm.shmimSat");

        m_shmimSatPerc = derived().m_shmimName + "SP";
        config(m_shmimSatPerc, "dm.shmimSatPerc");

        config(m_satAvgInt, "dm.satAvgInt");
    }
    else
    {
        config.isSet("dm.shmimFlat");
        config.isSet("dm.shmimTest");
        config.isSet("dm.shmimSat");
        config.isSet("dm.shmimSatPerc");
        config.isSet("dm.satAvgInt");
    }

    config(m_dmWidth, "dm.width");
    config(m_dmHeight, "dm.height");

    config(m_percThreshold, "dm.percThreshold");
    config(m_intervalSatThreshold, "dm.intervalSatThreshold");
    config(m_intervalSatCountThreshold, "dm.intervalSatCountThreshold");
    config(m_satTriggerDevice, "dm.satTriggerDevice");
    config(m_satTriggerProperty, "dm.satTriggerProperty");

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::appStartup()
{
    if (m_dmDataType == 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "unsupported DM data type"});
        return -1;
    }

    //-----------------
    // Get the flats
    checkFlats();

    // Register the test shmim INDI property
    m_indiP_flatShmim = pcf::IndiProperty(pcf::IndiProperty::Text);
    m_indiP_flatShmim.setDevice(derived().configName());
    m_indiP_flatShmim.setName("flat_shmim");
    m_indiP_flatShmim.setPerm(pcf::IndiProperty::ReadOnly);
    m_indiP_flatShmim.setState(pcf::IndiProperty::Idle);
    m_indiP_flatShmim.add(pcf::IndiElement("channel"));
    m_indiP_flatShmim["channel"] = m_shmimFlat;

    if (derived().registerIndiPropertyReadOnly(m_indiP_flatShmim) < 0)
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
        return -1;
    }

    // Register the setFlat INDI property
    derived().createStandardIndiToggleSw(m_indiP_setFlat, "flat_set");
    if (derived().registerIndiPropertyNew(m_indiP_setFlat, st_newCallBack_setFlat) < 0)
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
        return -1;
    }

    //-----------------
    // Get the tests
    checkTests();

    // Register the test shmim INDI property
    m_indiP_testShmim = pcf::IndiProperty(pcf::IndiProperty::Text);
    m_indiP_testShmim.setDevice(derived().configName());
    m_indiP_testShmim.setName("test_shmim");
    m_indiP_testShmim.setPerm(pcf::IndiProperty::ReadOnly);
    m_indiP_testShmim.setState(pcf::IndiProperty::Idle);
    m_indiP_testShmim.add(pcf::IndiElement("channel"));
    m_indiP_testShmim["channel"] = m_shmimTest;
    derived().createStandardIndiToggleSw(m_indiP_setTest, "test_shmim");
    if (derived().registerIndiPropertyReadOnly(m_indiP_testShmim) < 0)
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
        return -1;
    }

    // Register the setTest INDI property
    derived().createStandardIndiToggleSw(m_indiP_setTest, "test_set");
    if (derived().registerIndiPropertyNew(m_indiP_setTest, st_newCallBack_setTest) < 0)
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
        return -1;
    }

    // Register the init INDI property
    derived().createStandardIndiRequestSw(m_indiP_init, "initDM");
    if (derived().registerIndiPropertyNew(m_indiP_init, st_newCallBack_init) < 0)
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
        return -1;
    }

    // Register the zero INDI property
    derived().createStandardIndiRequestSw(m_indiP_zero, "zeroDM");
    if (derived().registerIndiPropertyNew(m_indiP_zero, st_newCallBack_zero) < 0)
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
        return -1;
    }

    // Register the release INDI property
    derived().createStandardIndiRequestSw(m_indiP_release, "releaseDM");
    if (derived().registerIndiPropertyNew(m_indiP_release, st_newCallBack_release) < 0)
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
        return -1;
    }

    derived().createStandardIndiRequestSw(m_indiP_zeroAll, "zeroAll");
    if (derived().registerIndiPropertyNew(m_indiP_zeroAll, st_newCallBack_zeroAll) < 0)
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
        return -1;
    }

    if (m_flatDefault != "")
    {
        loadFlat("default");
    }

    if (m_testDefault != "")
    {
        loadTest("default");
    }

    if (sem_init(&m_satSemaphore, 0, 0) < 0)
        return derivedT::template log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "Initializing sat semaphore"});

    if (derived().threadStart(m_satThread, m_satThreadInit, m_satThreadID, m_satThreadProp, m_satThreadPrio, "", "saturation", this, satThreadStart) < 0)
    {
        derivedT::template log<software_error, -1>({__FILE__, __LINE__});
        return -1;
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::appLogic()
{
    // do a join check to see if other threads have exited.
    if (pthread_tryjoin_np(m_satThread.native_handle(), 0) == 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "saturation thread has exited"});

        return -1;
    }

    checkFlats();
    checkTests();

    if (m_intervalSatTrip)
    {
        intervalSatTrip();
        m_intervalSatTrip = false;
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::appShutdown()
{
    if (m_satThread.joinable())
    {
        pthread_kill(m_satThread.native_handle(), SIGUSR1);
        try
        {
            m_satThread.join(); // this will throw if it was already joined
        }
        catch (...)
        {
        }
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::onPowerOff()
{
    releaseDM();
    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::whilePowerOff()
{
    checkFlats();
    checkTests();

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::findDMChannels()
{
    std::vector<std::string> dmlist = mx::ioutils::getFileNames("/milk/shm/", derived().m_shmimName, ".im", ".shm");

    if (dmlist.size() == 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "no dm channels found for " + derived().m_shmimName});
        return -1;
    }

    m_channels = -1;
    for (size_t n = 0; n < dmlist.size(); ++n)
    {
        char nstr[16];
        snprintf(nstr, sizeof(nstr), "%02d.im.shm", (int)n);
        std::string tgt = derived().m_shmimName;
        tgt += nstr;

        for (size_t m = 0; m < dmlist.size(); ++m)
        {
            if (dmlist[m].find(tgt) != std::string::npos)
            {
                if ((int)n > m_channels)
                    m_channels = n;
            }
        }
    }

    ++m_channels;

    derivedT::template log<text_log>({std::string("Found ") + std::to_string(m_channels) + " channels for " + derived().m_shmimName});

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::allocate(const dev::shmimT &sp)
{
    static_cast<void>(sp); // be unused

    int err = 0;

    if (derived().m_width != m_dmWidth)
    {
        derivedT::template log<software_critical>({__FILE__, __LINE__, "shmim width does not match configured DM width"});
        ++err;
    }

    if (derived().m_height != m_dmHeight)
    {
        derivedT::template log<software_critical>({__FILE__, __LINE__, "shmim height does not match configured DM height"});
        ++err;
    }

    if (derived().m_dataType != m_dmDataType)
    {
        derivedT::template log<software_critical>({__FILE__, __LINE__, "shmim data type does not match configured DM data type"});
        ++err;
    }

    if (err)
        return -1;

    m_instSatMap.resize(m_dmWidth, m_dmHeight);
    m_instSatMap.setZero();

    m_accumSatMap.resize(m_dmWidth, m_dmHeight);
    m_accumSatMap.setZero();

    m_satPercMap.resize(m_dmWidth, m_dmHeight);
    m_satPercMap.setZero();

    if (findDMChannels() < 0)
    {
        derivedT::template log<software_critical>({__FILE__, __LINE__, "error finding DM channels"});

        return -1;
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::processImage(void *curr_src,
                                      const dev::shmimT &sp)
{
    static_cast<void>(sp); // be unused

    int rv = derived().commandDM(curr_src);

    if (rv < 0)
    {
        derivedT::template log<software_critical>({__FILE__, __LINE__, errno, rv, "Error from commandDM"});
        return rv;
    }
    // Tell the sat thread to get going
    if (sem_post(&m_satSemaphore) < 0)
    {
        derivedT::template log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
        return -1;
    }

    return rv;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::releaseDM()
{
    int rv;
    if ((rv = derived().releaseDM()) < 0)
    {
        derivedT::template log<software_critical>({__FILE__, __LINE__, errno, rv, "Error from releaseDM"});
        return rv;
    }

    if ((rv = zeroAll(true)) < 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, errno, rv, "Error from zeroAll"});
        return rv;
    }

    return 0;
}
template <class derivedT, typename realT>
int dm<derivedT, realT>::checkFlats()
{
    std::vector<std::string> tfs = mx::ioutils::getFileNames(m_flatPath, "", "", ".fits");

    // First remove default, b/c we always add it and don't want to include it in timestamp selected ones
    for (size_t n = 0; n < tfs.size(); ++n)
    {
        if (mx::ioutils::pathStem(tfs[n]) == "default")
        {
            tfs.erase(tfs.begin() + n);
            --n;
        }
    }

    unsigned m_nFlatFiles = 5;

    // Here we keep only the m_nFlatFiles most recent files
    if (tfs.size() >= m_nFlatFiles)
    {
        std::vector<std::time_t> wtimes(tfs.size());

        for (size_t n = 0; n < wtimes.size(); ++n)
        {
            wtimes[n] = boost::filesystem::last_write_time(tfs[n]);
        }

        std::sort(wtimes.begin(), wtimes.end());

        std::time_t tn = wtimes[wtimes.size() - m_nFlatFiles];

        for (size_t n = 0; n < tfs.size(); ++n)
        {
            std::time_t lmt = boost::filesystem::last_write_time(tfs[n]);
            if (lmt < tn)
            {
                tfs.erase(tfs.begin() + n);
                --n;
            }
        }
    }

    for (auto it = m_flatCommands.begin(); it != m_flatCommands.end(); ++it)
    {
        it->second = "";
    }

    bool changed = false;
    for (size_t n = 0; n < tfs.size(); ++n)
    {
        auto ir = m_flatCommands.insert(std::pair<std::string, std::string>(mx::ioutils::pathStem(tfs[n]), tfs[n]));
        if (ir.second == true)
            changed = true;
        else
            ir.first->second = tfs[n];
    }

    for (auto it = m_flatCommands.begin(); it != m_flatCommands.end(); ++it)
    {
        if (it->second == "")
        {
            changed = true;
            // Erase the current iterator safely, even if the first one.
            auto itdel = it;
            ++it;
            m_flatCommands.erase(itdel);
            --it;
        };
    }

    if (changed)
    {
        if (derived().m_indiDriver)
        {
            derived().m_indiDriver->sendDelProperty(m_indiP_flats);
            derived().m_indiNewCallBacks.erase(m_indiP_flats.createUniqueKey());
        }

        m_indiP_flats = pcf::IndiProperty(pcf::IndiProperty::Switch);
        m_indiP_flats.setDevice(derived().configName());
        m_indiP_flats.setName("flat");
        m_indiP_flats.setPerm(pcf::IndiProperty::ReadWrite);
        m_indiP_flats.setState(pcf::IndiProperty::Idle);
        m_indiP_flats.setRule(pcf::IndiProperty::OneOfMany);

        // Add the toggle element initialized to Off
        for (auto it = m_flatCommands.begin(); it != m_flatCommands.end(); ++it)
        {
            if (it->first == m_flatCurrent || m_flatCurrent == "")
            {
                m_indiP_flats.add(pcf::IndiElement(it->first, pcf::IndiElement::On));
                m_flatCurrent = it->first; // handles the case m_flatCurrent == "" b/c it was not set in config
            }
            else
            {
                m_indiP_flats.add(pcf::IndiElement(it->first, pcf::IndiElement::Off));
            }
        }

        if (m_flatDefault != "")
        {
            if (m_flatCurrent == "default")
            {
                m_indiP_flats.add(pcf::IndiElement("default", pcf::IndiElement::On));
            }
            else
            {
                m_indiP_flats.add(pcf::IndiElement("default", pcf::IndiElement::Off));
            }
        }

        if (derived().registerIndiPropertyNew(m_indiP_flats, st_newCallBack_flats) < 0)
        {
#ifndef DM_TEST_NOLOG
            derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
            return -1;
        }

        if (derived().m_indiDriver)
        {
            derived().m_indiDriver->sendDefProperty(m_indiP_flats);
        }
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::loadFlat(const std::string &intarget)
{
    std::string target = intarget;

    std::string targetPath;

    if (target == "default")
    {
        target = m_flatDefault;
        targetPath = m_flatPath + "/" + m_flatDefault + ".fits";
    }
    else
    {
        try
        {
            targetPath = m_flatCommands.at(target);
        }
        catch (...)
        {
            derivedT::template log<text_log>("flat file " + target + " not found", logPrio::LOG_ERROR);
            return -1;
        }
    }

    m_flatLoaded = false;
    // load into memory.
    mx::fits::fitsFile<realT> ff;
    if (ff.read(m_flatCommand, targetPath) < 0)
    {
        derivedT::template log<text_log>("flat file " + targetPath + " not found", logPrio::LOG_ERROR);
        return -1;
    }

    derivedT::template log<text_log>("loaded flat file " + targetPath);
    m_flatLoaded = true;

    m_flatCurrent = intarget;

    if (m_indiP_flats.find("default"))
    {
        if (m_flatCurrent == "default")
        {
            m_indiP_flats["default"] = pcf::IndiElement::On;
        }
        else
        {
            m_indiP_flats["default"] = pcf::IndiElement::Off;
        }
    }

    for (auto i = m_flatCommands.begin(); i != m_flatCommands.end(); ++i)
    {
        if (!m_indiP_flats.find(i->first))
        {
            continue;
        }

        if (i->first == m_flatCurrent)
        {
            m_indiP_flats[i->first] = pcf::IndiElement::On;
        }
        else
        {
            m_indiP_flats[i->first] = pcf::IndiElement::Off;
        }
    }

    if (derived().m_indiDriver)
    {
        derived().m_indiDriver->sendSetProperty(m_indiP_flats);
    }

    if (m_flatSet)
    {
        setFlat();
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::setFlat(bool update)
{
    if (m_shmimFlat == "")
        return 0;

    if (ImageStreamIO_openIm(&m_flatImageStream, m_shmimFlat.c_str()) != 0)
    {
        derivedT::template log<text_log>("could not connect to flat channel " + m_shmimFlat, logPrio::LOG_WARNING);
        return -1;
    }

    if (m_flatImageStream.md[0].size[0] != m_dmWidth)
    {
        ImageStreamIO_closeIm(&m_flatImageStream);
        derivedT::template log<text_log>("width mismatch between " + m_shmimFlat + " and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_flatImageStream.md[0].size[1] != m_dmHeight)
    {
        ImageStreamIO_closeIm(&m_flatImageStream);
        derivedT::template log<text_log>("height mismatch between " + m_shmimFlat + " and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    if (!m_flatLoaded)
    {
        bool flatSet = m_flatSet;
        m_flatSet = false; // make sure we don't loop

        if (loadFlat(m_flatCurrent) < 0)
        {
            derivedT::template log<text_log>("error loading flat " + m_flatCurrent, logPrio::LOG_ERROR);
        }
        m_flatSet = flatSet;
    }

    if (!m_flatLoaded)
    {
        ImageStreamIO_closeIm(&m_flatImageStream);
        derivedT::template log<text_log>("no flat loaded", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_flatCommand.rows() != m_dmWidth)
    {
        ImageStreamIO_closeIm(&m_flatImageStream);
        derivedT::template log<text_log>("width mismatch between flat file and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_flatCommand.cols() != m_dmHeight)
    {
        ImageStreamIO_closeIm(&m_flatImageStream);
        derivedT::template log<text_log>("height mismatch between flat file and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    m_flatImageStream.md->write = 1;

    ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here anyway.  With bounds checks b/c not everyone handles cnt1 properly.
    // Copy
    memcpy(m_flatImageStream.array.raw, m_flatCommand.data(), m_dmWidth * m_dmHeight * sizeof(realT));

    // Set the time of last write
    clock_gettime(CLOCK_REALTIME, &m_flatImageStream.md->writetime);

    // Set the image acquisition timestamp
    m_flatImageStream.md->atime = m_flatImageStream.md->writetime;

    m_flatImageStream.md->cnt0++;
    m_flatImageStream.md->write = 0;
    ImageStreamIO_sempost(&m_flatImageStream, -1);

    m_flatSet = true;

    // Post the semaphore
    ImageStreamIO_closeIm(&m_flatImageStream);

    if (!update)
    {
        derived().updateSwitchIfChanged(m_indiP_setFlat, "toggle", pcf::IndiElement::On, pcf::IndiProperty::Busy);

        derivedT::template log<text_log>("flat set");
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::zeroFlat()
{
    if (m_shmimFlat == "")
        return 0;

    if (ImageStreamIO_openIm(&m_flatImageStream, m_shmimFlat.c_str()) != 0)
    {
        derivedT::template log<text_log>("could not connect to flat channel " + m_shmimFlat, logPrio::LOG_WARNING);
        return -1;
    }

    if (m_flatImageStream.md[0].size[0] != m_dmWidth)
    {
        ImageStreamIO_closeIm(&m_flatImageStream);
        derivedT::template log<text_log>("width mismatch between " + m_shmimFlat + " and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_flatImageStream.md[0].size[1] != m_dmHeight)
    {
        ImageStreamIO_closeIm(&m_flatImageStream);
        derivedT::template log<text_log>("height mismatch between " + m_shmimFlat + " and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    m_flatImageStream.md->write = 1;

    ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here anyway.  With bounds checks b/c not everyone handles cnt1 properly.
    // Zero
    memset(m_flatImageStream.array.raw, 0, m_dmWidth * m_dmHeight * sizeof(realT));

    // Set the time of last write
    clock_gettime(CLOCK_REALTIME, &m_flatImageStream.md->writetime);

    // Set the image acquisition timestamp
    m_flatImageStream.md->atime = m_flatImageStream.md->writetime;

    m_flatImageStream.md->cnt0++;
    m_flatImageStream.md->write = 0;
    ImageStreamIO_sempost(&m_flatImageStream, -1);

    m_flatSet = false;

    // Post the semaphore
    ImageStreamIO_closeIm(&m_flatImageStream);

    derived().updateSwitchIfChanged(m_indiP_setFlat, "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Idle);

    derivedT::template log<text_log>("flat zeroed");

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::checkTests()
{
    std::vector<std::string> tfs = mx::ioutils::getFileNames(m_testPath, "", "", ".fits");

    for (auto it = m_testCommands.begin(); it != m_testCommands.end(); ++it)
    {
        it->second = "";
    }

    bool changed = false;
    for (size_t n = 0; n < tfs.size(); ++n)
    {
        auto ir = m_testCommands.insert(std::pair<std::string, std::string>(mx::ioutils::pathStem(tfs[n]), tfs[n]));
        if (ir.second == true)
            changed = true;
        else
            ir.first->second = tfs[n];
    }

    for (auto it = m_testCommands.begin(); it != m_testCommands.end(); ++it)
    {
        if (it->second == "")
        {
            changed = true;
            // Erase the current iterator safely, even if the first one.
            auto itdel = it;
            ++it;
            m_testCommands.erase(itdel);
            --it;
        };
    }

    if (changed)
    {
        if (derived().m_indiDriver)
        {
            derived().m_indiDriver->sendDelProperty(m_indiP_tests);
            derived().m_indiNewCallBacks.erase(m_indiP_tests.createUniqueKey());
        }

        m_indiP_tests = pcf::IndiProperty(pcf::IndiProperty::Switch);
        m_indiP_tests.setDevice(derived().configName());
        m_indiP_tests.setName("test");
        m_indiP_tests.setPerm(pcf::IndiProperty::ReadWrite);
        m_indiP_tests.setState(pcf::IndiProperty::Idle);
        m_indiP_tests.setRule(pcf::IndiProperty::OneOfMany);

        // Add the toggle element initialized to Off
        for (auto it = m_testCommands.begin(); it != m_testCommands.end(); ++it)
        {
            if (it->first == m_testCurrent || m_testCurrent == "")
            {
                m_indiP_tests.add(pcf::IndiElement(it->first, pcf::IndiElement::On));
                m_testCurrent = it->first; // Handles the case when m_testCurrent=="" b/c it was not set in config
            }
            else
            {
                m_indiP_tests.add(pcf::IndiElement(it->first, pcf::IndiElement::Off));
            }
        }

        if (m_testDefault != "")
        {
            if (m_testCurrent == "default")
            {
                m_indiP_tests.add(pcf::IndiElement("default", pcf::IndiElement::On));
            }
            else
            {
                m_indiP_tests.add(pcf::IndiElement("default", pcf::IndiElement::Off));
            }
        }

        if (derived().registerIndiPropertyNew(m_indiP_tests, st_newCallBack_tests) < 0)
        {
#ifndef DM_TEST_NOLOG
            derivedT::template log<software_error>({__FILE__, __LINE__});
#endif
            return -1;
        }

        if (derived().m_indiDriver)
        {
            derived().m_indiDriver->sendDefProperty(m_indiP_tests);
        }
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::loadTest(const std::string &intarget)
{
    std::string target = intarget; // store this for later to resolve default next:

    if (target == "default")
    {
        target = m_testDefault;
    }

    std::string targetPath;

    try
    {
        targetPath = m_testCommands.at(target);
    }
    catch (...)
    {
        derivedT::template log<text_log>("test file " + target + " not found", logPrio::LOG_ERROR);
        return -1;
    }

    m_testLoaded = false;
    // load into memory.
    mx::fits::fitsFile<realT> ff;
    if (ff.read(m_testCommand, targetPath) < 0)
    {
        derivedT::template log<text_log>("test file " + targetPath + " not found", logPrio::LOG_ERROR);
        return -1;
    }

    derivedT::template log<text_log>("loaded test file " + targetPath);
    m_testLoaded = true;

    m_testCurrent = intarget;

    if (m_indiP_tests.find("default"))
    {
        if (m_testCurrent == "default")
        {
            m_indiP_tests["default"] = pcf::IndiElement::On;
        }
        else
        {
            m_indiP_tests["default"] = pcf::IndiElement::Off;
        }
    }

    for (auto i = m_testCommands.begin(); i != m_testCommands.end(); ++i)
    {
        if (!m_indiP_tests.find(i->first))
        {
            continue;
        }

        if (i->first == m_testCurrent)
        {
            m_indiP_tests[i->first] = pcf::IndiElement::On;
        }
        else
        {
            m_indiP_tests[i->first] = pcf::IndiElement::Off;
        }
    }

    if (derived().m_indiDriver)
        derived().m_indiDriver->sendSetProperty(m_indiP_tests);

    if (m_testSet)
        setTest();

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::setTest()
{

    if (m_shmimTest == "")
        return 0;

    if (ImageStreamIO_openIm(&m_testImageStream, m_shmimTest.c_str()) != 0)
    {
        derivedT::template log<text_log>("could not connect to test channel " + m_shmimTest, logPrio::LOG_WARNING);
        return -1;
    }

    if (m_testImageStream.md->size[0] != m_dmWidth)
    {
        ImageStreamIO_closeIm(&m_testImageStream);
        derivedT::template log<text_log>("width mismatch between " + m_shmimTest + " and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_testImageStream.md->size[1] != m_dmHeight)
    {
        ImageStreamIO_closeIm(&m_testImageStream);
        derivedT::template log<text_log>("height mismatch between " + m_shmimTest + " and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    if (!m_testLoaded)
    {
        bool testSet = m_testSet;
        m_testSet = false; // make sure we don't loop

        if (loadTest(m_testCurrent) < 0)
        {
            derivedT::template log<text_log>("error loading test " + m_testCurrent, logPrio::LOG_ERROR);
        }
        m_testSet = testSet;
    }

    if (!m_testLoaded)
    {
        ImageStreamIO_closeIm(&m_testImageStream);
        derivedT::template log<text_log>("no test loaded", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_testCommand.rows() != m_dmWidth)
    {
        ImageStreamIO_closeIm(&m_testImageStream);
        derivedT::template log<text_log>("width mismatch between test file and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_testCommand.cols() != m_dmHeight)
    {
        ImageStreamIO_closeIm(&m_testImageStream);
        derivedT::template log<text_log>("height mismatch between test file and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    m_testImageStream.md->write = 1;

    ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here anyway.  With bounds checks b/c not everyone handles cnt1 properly.
    // Copy
    memcpy(m_testImageStream.array.raw, m_testCommand.data(), m_dmWidth * m_dmHeight * sizeof(realT));

    // Set the time of last write
    clock_gettime(CLOCK_REALTIME, &m_testImageStream.md->writetime);

    // Set the image acquisition timestamp
    m_testImageStream.md->atime = m_testImageStream.md->writetime;

    m_testImageStream.md->cnt0++;
    m_testImageStream.md->write = 0;
    ImageStreamIO_sempost(&m_testImageStream, -1);

    m_testSet = true;

    // Post the semaphore
    ImageStreamIO_closeIm(&m_testImageStream);

    derived().updateSwitchIfChanged(m_indiP_setTest, "toggle", pcf::IndiElement::On, pcf::IndiProperty::Busy);

    derivedT::template log<text_log>("test set");

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::zeroTest()
{
    if (m_shmimTest == "")
        return 0;

    if (ImageStreamIO_openIm(&m_testImageStream, m_shmimTest.c_str()) != 0)
    {
        derivedT::template log<text_log>("could not connect to test channel " + m_shmimTest, logPrio::LOG_WARNING);
        return -1;
    }

    if (m_testImageStream.md[0].size[0] != m_dmWidth)
    {
        ImageStreamIO_closeIm(&m_testImageStream);
        derivedT::template log<text_log>("width mismatch between " + m_shmimTest + " and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_testImageStream.md[0].size[1] != m_dmHeight)
    {
        ImageStreamIO_closeIm(&m_testImageStream);
        derivedT::template log<text_log>("height mismatch between " + m_shmimTest + " and configured DM", logPrio::LOG_ERROR);
        return -1;
    }

    m_testImageStream.md->write = 1;

    ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here anyway.  With bounds checks b/c not everyone handles cnt1 properly.
    // Zero
    memset(m_testImageStream.array.raw, 0, m_dmWidth * m_dmHeight * sizeof(realT));

    // Set the time of last write
    clock_gettime(CLOCK_REALTIME, &m_testImageStream.md->writetime);

    // Set the image acquisition timestamp
    m_testImageStream.md->atime = m_testImageStream.md->writetime;

    m_testImageStream.md->cnt0++;
    m_testImageStream.md->write = 0;

    // Post the semaphore
    ImageStreamIO_sempost(&m_testImageStream, -1);

    m_testSet = false;

    ImageStreamIO_closeIm(&m_testImageStream);

    derived().updateSwitchIfChanged(m_indiP_setTest, "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Idle);

    derivedT::template log<text_log>("test zeroed");

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::zeroAll(bool nosem)
{
    if (derived().m_shmimName == "")
        return 0;

    IMAGE imageStream;

    for (int n = 0; n < m_channels; ++n)
    {
        char nstr[16];
        snprintf(nstr, sizeof(nstr), "%02d", n);
        std::string shmimN = derived().m_shmimName + nstr;

        if (ImageStreamIO_openIm(&imageStream, shmimN.c_str()) != 0)
        {
            derivedT::template log<text_log>("could not connect to channel " + shmimN, logPrio::LOG_WARNING);
            continue;
        }

        if (imageStream.md->size[0] != m_dmWidth)
        {
            ImageStreamIO_closeIm(&imageStream);
            derivedT::template log<text_log>("width mismatch between " + shmimN + " and configured DM", logPrio::LOG_ERROR);
            derived().updateSwitchIfChanged(m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE);
            return -1;
        }

        if (imageStream.md->size[1] != m_dmHeight)
        {
            ImageStreamIO_closeIm(&imageStream);
            derivedT::template log<text_log>("height mismatch between " + shmimN + " and configured DM", logPrio::LOG_ERROR);
            derived().updateSwitchIfChanged(m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE);
            return -1;
        }

        imageStream.md->write = 1;
        memset(imageStream.array.raw, 0, m_dmWidth * m_dmHeight * sizeof(realT));

        clock_gettime(CLOCK_REALTIME, &imageStream.md->writetime);

        // Set the image acquisition timestamp
        imageStream.md->atime = imageStream.md->writetime;

        imageStream.md->cnt0++;
        imageStream.md->write = 0;

        // Raise the semaphore on last one.
        if (n == m_channels - 1 && !nosem)
            ImageStreamIO_sempost(&imageStream, -1);

        ImageStreamIO_closeIm(&imageStream);
    }

    derivedT::template log<text_log>("all channels zeroed", logPrio::LOG_NOTICE);

    derived().updateSwitchIfChanged(m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE);

    // Also cleanup flat and test
    m_flatSet = false;
    derived().updateSwitchIfChanged(m_indiP_setFlat, "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Idle);

    // Also cleanup flat and test
    m_testSet = false;
    derived().updateSwitchIfChanged(m_indiP_setTest, "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Idle);

    int rv;
    if ((rv = clearSat()) < 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, errno, rv, "Error from clearSat"});
        return rv;
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::clearSat()
{
    if (m_shmimSat == "")
        return 0;

    IMAGE imageStream;

    std::vector<std::string> sats = {m_shmimSat, m_shmimSatPerc};

    for (size_t n = 0; n < sats.size(); ++n)
    {
        std::string shmimN = sats[n];

        if (ImageStreamIO_openIm(&imageStream, shmimN.c_str()) != 0)
        {
            derivedT::template log<text_log>("could not connect to sat map " + shmimN, logPrio::LOG_WARNING);
            return 0;
        }

        if (imageStream.md->size[0] != m_dmWidth)
        {
            ImageStreamIO_closeIm(&imageStream);
            derivedT::template log<text_log>("width mismatch between " + shmimN + " and configured DM", logPrio::LOG_ERROR);
            derived().updateSwitchIfChanged(m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE);
            return -1;
        }

        if (imageStream.md->size[1] != m_dmHeight)
        {
            ImageStreamIO_closeIm(&imageStream);
            derivedT::template log<text_log>("height mismatch between " + shmimN + " and configured DM", logPrio::LOG_ERROR);
            derived().updateSwitchIfChanged(m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE);
            return -1;
        }

        imageStream.md->write = 1;
        memset(imageStream.array.raw, 0, m_dmWidth * m_dmHeight * sizeof(realT));

        clock_gettime(CLOCK_REALTIME, &imageStream.md->writetime);

        // Set the image acquisition timestamp
        imageStream.md->atime = imageStream.md->writetime;

        imageStream.md->cnt0++;
        imageStream.md->write = 0;

        ImageStreamIO_closeIm(&imageStream);
    }

    m_accumSatMap.setZero();
    m_instSatMap.setZero();

    return 0;
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::satThreadStart(dm *d)
{
    d->satThreadExec();
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::satThreadExec()
{
    // Get the thread PID immediately so the caller can return.
    m_satThreadID = syscall(SYS_gettid);

    // Wait for the thread starter to finish initializing this thread.
    while (m_satThreadInit == true && derived().shutdown() == 0)
    {
        sleep(1);
    }
    if (derived().shutdown())
        return;

    uint32_t imsize[3] = {0, 0, 0};

    // Check for allocation to have happened.
    while ((m_shmimSat == "" || m_accumSatMap.rows() == 0 || m_accumSatMap.cols() == 0) && !derived().shutdown())
    {
        sleep(1);
    }
    if (derived().shutdown())
        return;

    imsize[0] = m_dmWidth;
    imsize[1] = m_dmHeight;
    imsize[2] = 1;

    ImageStreamIO_createIm_gpu(&m_satImageStream, m_shmimSat.c_str(), 3, imsize, IMAGESTRUCT_UINT8, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0);
    ImageStreamIO_createIm_gpu(&m_satPercImageStream, m_shmimSatPerc.c_str(), 3, imsize, IMAGESTRUCT_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0);

    bool opened = true;

    m_satImageStream.md->cnt1 = 0;
    m_satPercImageStream.md->cnt1 = 0;

    // This is the working memory for making the 1/0 mask out of m_accumSatMap
    mx::improc::eigenImage<uint8_t> satmap(m_dmWidth, m_dmHeight);

    int naccum = 0;
    double t_accumst = mx::sys::get_curr_time();

    // This is the main image grabbing loop.
    while (!derived().shutdown())
    {
        // Get timespec for sem_timedwait
        timespec ts;
        if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
        {
            derivedT::template log<software_critical>({__FILE__, __LINE__, errno, 0, "clock_gettime"});
            return;
        }
        ts.tv_sec += 1;

        // Wait on semaphore
        if (sem_timedwait(&m_satSemaphore, &ts) == 0)
        {
            // not a timeout -->accumulate
            for (int rr = 0; rr < m_instSatMap.rows(); ++rr)
            {
                for (int cc = 0; cc < m_instSatMap.cols(); ++cc)
                {
                    m_accumSatMap(rr, cc) += m_instSatMap(rr, cc);
                }
            }
            ++naccum;

            // If less than avg int --> go back and wait again
            if (mx::sys::get_curr_time(ts) - t_accumst < m_satAvgInt / 1000.0)
                continue;

            // If greater than avg int --> calc stats, write to streams.
            m_overSatAct = 0;
            for (int rr = 0; rr < m_instSatMap.rows(); ++rr)
            {
                for (int cc = 0; cc < m_instSatMap.cols(); ++cc)
                {
                    m_satPercMap(rr, cc) = m_accumSatMap(rr, cc) / naccum;
                    if (m_satPercMap(rr, cc) >= m_percThreshold)
                        ++m_overSatAct;
                    satmap(rr, cc) = (m_accumSatMap(rr, cc) > 0); // it's  1/0 map
                }
            }

            // Check of the number of actuators saturated above the percent threshold is greater than the number threshold
            // if it is, increment the counter
            if (m_overSatAct / (m_satPercMap.rows() * m_satPercMap.cols() * 0.75) > m_intervalSatThreshold)
                ++m_intervalSatExceeds;
            else
                m_intervalSatExceeds = 0;

            // If enough consecutive intervals exceed the count threshold, we trigger
            if (m_intervalSatExceeds >= m_intervalSatCountThreshold)
                m_intervalSatTrip = true;

            m_satImageStream.md->write = 1;
            m_satPercImageStream.md->write = 1;

            memcpy(m_satImageStream.array.raw, satmap.data(), m_dmWidth * m_dmHeight * sizeof(uint8_t));
            memcpy(m_satPercImageStream.array.raw, m_satPercMap.data(), m_dmWidth * m_dmHeight * sizeof(float));

            // Set the time of last write
            clock_gettime(CLOCK_REALTIME, &m_satImageStream.md->writetime);
            m_satPercImageStream.md->writetime = m_satImageStream.md->writetime;

            // Set the image acquisition timestamp
            m_satImageStream.md->atime = m_satImageStream.md->writetime;
            m_satPercImageStream.md->atime = m_satPercImageStream.md->writetime;

            // Update cnt1
            m_satImageStream.md->cnt1 = 0;
            m_satPercImageStream.md->cnt1 = 0;

            // Update cnt0
            m_satImageStream.md->cnt0++;
            m_satPercImageStream.md->cnt0++;

            m_satImageStream.writetimearray[0] = m_satImageStream.md->writetime;
            m_satImageStream.atimearray[0] = m_satImageStream.md->atime;
            m_satImageStream.cntarray[0] = m_satImageStream.md->cnt0;

            m_satPercImageStream.writetimearray[0] = m_satPercImageStream.md->writetime;
            m_satPercImageStream.atimearray[0] = m_satPercImageStream.md->atime;
            m_satPercImageStream.cntarray[0] = m_satPercImageStream.md->cnt0;

            // And post
            m_satImageStream.md->write = 0;
            ImageStreamIO_sempost(&m_satImageStream, -1);

            m_satPercImageStream.md->write = 0;
            ImageStreamIO_sempost(&m_satPercImageStream, -1);

            m_accumSatMap.setZero();
            naccum = 0;
            t_accumst = mx::sys::get_curr_time(ts);
        }
        else
        {
            // Check for why we timed out
            if (errno == EINTR)
                break; // This indicates signal interrupted us, time to restart or shutdown, loop will exit normally if flags set.

            // ETIMEDOUT just means we should wait more.
            // Otherwise, report an error.
            if (errno != ETIMEDOUT)
            {
                derivedT::template log<software_error>({__FILE__, __LINE__, errno, "sem_timedwait"});
                break;
            }
        }
    }

    if (opened)
    {
        ImageStreamIO_destroyIm(&m_satImageStream);

        ImageStreamIO_destroyIm(&m_satPercImageStream);
    }
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::updateINDI()
{
    if (!derived().m_indiDriver)
        return 0;

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_init(void *app,
                                             const pcf::IndiProperty &ipRecv)
{
    return static_cast<derivedT *>(app)->newCallBack_init(ipRecv);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_init(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() != m_indiP_init.createUniqueKey())
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
    }

    if (!ipRecv.find("request"))
        return 0;

    if (ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
        return derived().initDM();
    }
    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_zero(void *app,
                                             const pcf::IndiProperty &ipRecv)
{
    return static_cast<derivedT *>(app)->newCallBack_zero(ipRecv);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_zero(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() != m_indiP_zero.createUniqueKey())
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
    }

    if (!ipRecv.find("request"))
        return 0;

    if (ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
        return derived().zeroDM();
    }
    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_release(void *app,
                                                const pcf::IndiProperty &ipRecv)
{
    return static_cast<derivedT *>(app)->newCallBack_release(ipRecv);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_release(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() != m_indiP_release.createUniqueKey())
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
    }

    if (!ipRecv.find("request"))
        return 0;

    if (ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
        return releaseDM();
    }
    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_flats(void *app,
                                              const pcf::IndiProperty &ipRecv)
{
    return static_cast<derivedT *>(app)->newCallBack_flats(ipRecv);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_flats(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() != m_indiP_flats.createUniqueKey())
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
        return -1;
    }

    std::string newFlat;

    if (ipRecv.find("default"))
    {
        if (ipRecv["default"].getSwitchState() == pcf::IndiElement::On)
        {
            newFlat = "default";
        }
    }

    // always do this to check for error:
    for (auto i = m_flatCommands.begin(); i != m_flatCommands.end(); ++i)
    {
        if (!ipRecv.find(i->first))
            continue;

        if (ipRecv[i->first].getSwitchState() == pcf::IndiElement::On)
        {
            if (newFlat != "")
            {
                derivedT::template log<text_log>("More than one flat selected", logPrio::LOG_ERROR);
                return -1;
            }

            newFlat = i->first;
        }
    }

    if (newFlat == "")
        return 0;

    return loadFlat(newFlat);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_setFlat(void *app,
                                                const pcf::IndiProperty &ipRecv)
{
    return static_cast<derivedT *>(app)->newCallBack_setFlat(ipRecv);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_setFlat(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() != m_indiP_setFlat.createUniqueKey())
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
    }

    if (!ipRecv.find("toggle"))
        return 0;

    if (ipRecv["toggle"] == pcf::IndiElement::On)
    {
        return setFlat();
    }
    else
    {
        return zeroFlat();
    }
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_tests(void *app,
                                              const pcf::IndiProperty &ipRecv)
{
    return static_cast<derivedT *>(app)->newCallBack_tests(ipRecv);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_tests(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() != m_indiP_tests.createUniqueKey())
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
        return -1;
    }

    std::string newTest;

    if (ipRecv.find("default"))
    {
        if (ipRecv["default"].getSwitchState() == pcf::IndiElement::On)
        {
            newTest = "default";
        }
    }

    // always do this to check for error:
    for (auto i = m_testCommands.begin(); i != m_testCommands.end(); ++i)
    {
        if (!ipRecv.find(i->first))
            continue;

        if (ipRecv[i->first].getSwitchState() == pcf::IndiElement::On)
        {
            if (newTest != "")
            {
                derivedT::template log<text_log>("More than one test selected", logPrio::LOG_ERROR);
                return -1;
            }

            newTest = i->first;
        }
    }

    if (newTest == "")
        return 0;

    return loadTest(newTest);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_setTest(void *app,
                                                const pcf::IndiProperty &ipRecv)
{
    return static_cast<derivedT *>(app)->newCallBack_setTest(ipRecv);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_setTest(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() != m_indiP_setTest.createUniqueKey())
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
    }

    if (!ipRecv.find("toggle"))
        return 0;

    if (ipRecv["toggle"] == pcf::IndiElement::On)
    {
        return setTest();
    }
    else
    {
        return zeroTest();
    }
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_zeroAll(void *app,
                                                const pcf::IndiProperty &ipRecv)
{
    return static_cast<derivedT *>(app)->newCallBack_zeroAll(ipRecv);
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_zeroAll(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() != m_indiP_zeroAll.createUniqueKey())
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
    }

    if (!ipRecv.find("request"))
        return 0;

    if (ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
        indi::updateSwitchIfChanged(m_indiP_zeroAll, "request", pcf::IndiElement::On, derived().m_indiDriver, INDI_BUSY);

        std::lock_guard<std::mutex> guard(derived().m_indiMutex);
        return zeroAll();
    }
    return 0;
}

/// Call dmT::setupConfig with error checking for dm
/**
 * \param cfig the application configurator
 */
#define DM_SETUP_CONFIG(cfig)                                                     \
    if (dmT::setupConfig(cfig) < 0)                                               \
    {                                                                             \
        log<software_error>({__FILE__, __LINE__, "Error from dmT::setupConfig"}); \
        m_shutdown = true;                                                        \
        return;                                                                   \
    }

/// Call dmT::loadConfig with error checking for dm
/** This must be inside a function that returns int, e.g. the standard loadConfigImpl.
 * \param cfig the application configurator
 */
#define DM_LOAD_CONFIG(cfig)                                                                \
    if (dmT::loadConfig(cfig) < 0)                                                          \
    {                                                                                       \
        return log<software_error, -1>({__FILE__, __LINE__, "Error from dmT::loadConfig"}); \
    }

/// Call shmimMonitorT::appStartup with error checking for dm
#define DM_APP_STARTUP                                                                      \
    if (dmT::appStartup() < 0)                                                              \
    {                                                                                       \
        return log<software_error, -1>({__FILE__, __LINE__, "Error from dmT::appStartup"}); \
    }

/// Call dmT::appLogic with error checking for dm
#define DM_APP_LOGIC                                                                      \
    if (dmT::appLogic() < 0)                                                              \
    {                                                                                     \
        return log<software_error, -1>({__FILE__, __LINE__, "Error from dmT::appLogic"}); \
    }

/// Call dmT::updateINDI with error checking for dm
#define DM_UPDATE_INDI                                                                      \
    if (dmT::updateINDI() < 0)                                                              \
    {                                                                                       \
        return log<software_error, -1>({__FILE__, __LINE__, "Error from dmT::updateINDI"}); \
    }

/// Call dmT::appShutdown with error checking for dm
#define DM_APP_SHUTDOWN                                                                      \
    if (dmT::appShutdown() < 0)                                                              \
    {                                                                                        \
        return log<software_error, -1>({__FILE__, __LINE__, "Error from dmT::appShutdown"}); \
    }

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
