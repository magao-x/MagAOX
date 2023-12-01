/** \file cameraSim_test.cpp
  * \brief Catch2 tests for the cameraSim app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../cameraSim.hpp"

using namespace MagAOX::app;

namespace SMCTEST
{

class cameraSim_test : public cameraSim 
{

public:
    cameraSim_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, reconfigure)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, temp_ccd )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, temp_controller)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, readout_speed)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, vshift_speed)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, emgain)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, exptime)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, fps)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, synchro)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, mode)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_crop_mode)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_region_x)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_region_y)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_region_w)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_region_h)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_region_bin_x)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_region_bin_y)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_region_check)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_set)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_set_full)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_set_full_bin)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_load_last)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_set_last)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, roi_set_default)
        XWCTEST_SETUP_INDI_ARB_NEW_PROP(m_indiP_temp, shutter)
    }
};


SCENARIO( "INDI Callbacks", "[cameraSim]" )
{
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, reconfigure);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, temp_ccd);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, temp_controller);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, readout_speed);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, vshift_speed);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, emgain);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, exptime);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, fps);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, synchro);
    //XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, mode);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_crop_mode);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_region_x);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_region_y);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_region_w);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_region_h);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_region_bin_x);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_region_bin_y);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_region_check);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_set);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_set_full);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_set_full_bin);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_load_last);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_set_last);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, roi_set_default);
    XWCTEST_INDI_ARBNEW_CALLBACK(cameraSim, newCallBack_stdCamera, shutter);
}


} //namespace cameraSim_test 
