/** \file userGainCtrl_test.cpp
  * \brief Catch2 tests for the userGainCtrl app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../userGainCtrl.hpp"

using namespace MagAOX::app;

namespace SMCTEST
{

class userGainCtrl_test : public userGainCtrl 
{

public:
    userGainCtrl_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(zeroAll);
        XWCTEST_SETUP_INDI_NEW_PROP(singleModeNo);
        XWCTEST_SETUP_INDI_NEW_PROP(singleGain);
        XWCTEST_SETUP_INDI_NEW_PROP(singleMC);
    
    }
};

SCENARIO( "INDI Callbacks", "[userGainCtrl]" )
{
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, zeroAll);
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, singleModeNo);
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, singleGain);
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, singleMC);
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_blockGains, block00_gain);
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_blockMCs, block70_multcoeff);
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_blockLimits, block32_limit);

}

SCENARIO( "Calculating Blocks", "[userGainCtrl]" )
{
    GIVEN("No Zernikes")
    {
        int rv;
        std::vector<uint16_t> blocks;
        std::vector<std::string> names;
        //A full block means that all the modes in the last block are present
        WHEN("Full blocks")
        {
            rv = blockModes(blocks, names, 24, 0, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 1);
            REQUIRE(blocks[0] == 24);

            rv = blockModes(blocks, names, 80, 0, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 2);
            REQUIRE(blocks[0] == 24);
            REQUIRE(blocks[1] == 56);

            rv = blockModes(blocks, names, 168, 0, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 3);
            REQUIRE(blocks[0] == 24);
            REQUIRE(blocks[1] == 56);
            REQUIRE(blocks[2] == 88);
        }

        //A partial block means that not all modes in last block are present
        WHEN("Partial blocks")
        {
            rv = blockModes(blocks, names, 25, 0, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 2);
            REQUIRE(blocks[0] == 24);
            REQUIRE(blocks[1] == 1);

            rv = blockModes(blocks, names, 85, 0, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 3);
            REQUIRE(blocks[0] == 24);
            REQUIRE(blocks[1] == 56);
            REQUIRE(blocks[2] == 5);

            rv = blockModes(blocks, names, 287, 0, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 4);
            REQUIRE(blocks[0] == 24);
            REQUIRE(blocks[1] == 56);
            REQUIRE(blocks[2] == 88);
            REQUIRE(blocks[3] == 119);

        }

        //Test when Zernikes cover various numbers of blocks
        WHEN("Full blocks, Zernikes in 1 block, no split")
        {
            rv = blockModes(blocks, names, 24, 10, false);
            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 4);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 7);
            REQUIRE(blocks[3] == 14);

            
            rv = blockModes(blocks, names, 80, 5, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 5);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 2);
            REQUIRE(blocks[3] == 19);
            REQUIRE(blocks[4] == 56);

            rv = blockModes(blocks, names, 168, 23, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 6);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 20);
            REQUIRE(blocks[3] == 1);
            REQUIRE(blocks[4] == 56);
            REQUIRE(blocks[5] == 88);

        }

        WHEN("Full blocks, Zernikes in 2 blocks, no split")
        {
            rv = blockModes(blocks, names, 24, 25, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 3);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 22);

            rv = blockModes(blocks, names, 80, 25, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 4);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 22);
            REQUIRE(blocks[3] == 55);

            rv = blockModes(blocks, names, 168, 79, false);
            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 5);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 76);
            REQUIRE(blocks[3] == 1);
            REQUIRE(blocks[4] == 88);

        }

        WHEN("Partial blocks, Zernikes in 1 block, no split")
        {
            rv = blockModes(blocks, names, 25, 10, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 5);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 7);
            REQUIRE(blocks[3] == 14);
            REQUIRE(blocks[4] == 1);

            rv = blockModes(blocks, names, 85, 5, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 6);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 2);
            REQUIRE(blocks[3] == 19);
            REQUIRE(blocks[4] == 56);
            REQUIRE(blocks[5] == 5);

            rv = blockModes(blocks, names, 287, 23, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 7);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 20);
            REQUIRE(blocks[3] == 1);
            REQUIRE(blocks[4] == 56);
            REQUIRE(blocks[5] == 88);
            REQUIRE(blocks[6] == 119);

        }

        WHEN("Partial blocks, Zernikes in 2 blocks, no split")
        {
            rv = blockModes(blocks, names, 26, 25, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 4);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 22);
            REQUIRE(blocks[3] == 1);

            rv = blockModes(blocks, names, 85, 25, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 5);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 22);
            REQUIRE(blocks[3] == 55);
            REQUIRE(blocks[4] == 5);

            rv = blockModes(blocks, names, 287, 79, false);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 6);
            REQUIRE(blocks[0] == 2);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 76);
            REQUIRE(blocks[3] == 1); //b = 1, 80
            REQUIRE(blocks[4] == 88); //b = 2, 168
            REQUIRE(blocks[5] == 119);

        }

        WHEN("Full blocks, Zernikes in 1 block, with split")
        {
            rv = blockModes(blocks, names, 24, 10, true);
            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 5);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 7);
            REQUIRE(blocks[4] == 14);

            
            rv = blockModes(blocks, names, 80, 5, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 6);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 2);
            REQUIRE(blocks[4] == 19);
            REQUIRE(blocks[5] == 56);

            rv = blockModes(blocks, names, 168, 23, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 7);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 20);
            REQUIRE(blocks[4] == 1);
            REQUIRE(blocks[5] == 56);
            REQUIRE(blocks[6] == 88);

        }

        WHEN("Full blocks, Zernikes in 2 blocks, with split")
        {
            rv = blockModes(blocks, names, 24, 25, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 4);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 22);

            rv = blockModes(blocks, names, 80, 25, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 5);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 22);
            REQUIRE(blocks[4] == 55);

            rv = blockModes(blocks, names, 168, 79, true);
            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 6);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 76);
            REQUIRE(blocks[4] == 1);
            REQUIRE(blocks[5] == 88);

        }

        WHEN("Partial blocks, Zernikes in 1 block, with split")
        {
            rv = blockModes(blocks, names, 25, 10, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 6);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 7);
            REQUIRE(blocks[4] == 14);
            REQUIRE(blocks[5] == 1);

            rv = blockModes(blocks, names, 85, 5, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 7);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 2);
            REQUIRE(blocks[4] == 19);
            REQUIRE(blocks[5] == 56);
            REQUIRE(blocks[6] == 5);

            rv = blockModes(blocks, names, 287, 23, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 8);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 20);
            REQUIRE(blocks[4] == 1);
            REQUIRE(blocks[5] == 56);
            REQUIRE(blocks[6] == 88);
            REQUIRE(blocks[7] == 119);

        }

        WHEN("Partial blocks, Zernikes in 2 blocks, with split")
        {
            rv = blockModes(blocks, names, 26, 25, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 5);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 22);
            REQUIRE(blocks[4] == 1);

            rv = blockModes(blocks, names, 85, 25, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 6);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 22);
            REQUIRE(blocks[4] == 55);
            REQUIRE(blocks[5] == 5);

            rv = blockModes(blocks, names, 287, 79, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 7);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 76);
            REQUIRE(blocks[4] == 1); //b = 1, 80
            REQUIRE(blocks[5] == 88); //b = 2, 168
            REQUIRE(blocks[6] == 119);

        }

        //Test if it all generalizes to even more zernikes
        WHEN("Partial blocks, Zernikes in 3 blocks, with split")
        {
            rv = blockModes(blocks, names, 26, 81, true); //this should give 81 modes and quit

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 4);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 78);
 
            rv = blockModes(blocks, names, 85, 81, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 5);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 78);
            REQUIRE(blocks[4] == 4);

            rv = blockModes(blocks, names, 287, 100, true);

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 6);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 97);
            REQUIRE(blocks[4] == 68); //b = 2, 168
            REQUIRE(blocks[5] == 119);

        }

        WHEN("The Full MagAO-X")
        {
            rv = blockModes(blocks, names, 2400, 10, true); //this should give 81 modes and quite

            REQUIRE(rv == 0);
            REQUIRE(blocks.size() == 16);
            REQUIRE(blocks[0] == 1);
            REQUIRE(blocks[1] == 1);
            REQUIRE(blocks[2] == 1);
            REQUIRE(blocks[3] == 7);
            REQUIRE(blocks[4] == 14); //b=0 remainder
            REQUIRE(blocks[5] == 56); //b=1
            REQUIRE(blocks[6] == 88);
            REQUIRE(blocks[7] == 120);
            REQUIRE(blocks[8] == 152);
            REQUIRE(blocks[9] == 184);
            REQUIRE(blocks[10] == 216);
            REQUIRE(blocks[11] == 248);
            REQUIRE(blocks[12] == 280);
            REQUIRE(blocks[13] == 312);
            REQUIRE(blocks[14] == 344);
            REQUIRE(blocks[15] == 376);
        }
    }
}


} //namespace userGainCtrl_test 
