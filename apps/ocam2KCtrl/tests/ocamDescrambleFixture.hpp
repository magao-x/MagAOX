/** \file ocamDescrambleFixture.hpp
 * \brief Deterministic OCAM amplifier mapping for hardware-free descramble tests.
 */

#ifndef ocam2KCtrl_tests_ocamDescrambleFixture_hpp
#define ocam2KCtrl_tests_ocamDescrambleFixture_hpp

#include <vector>

namespace libXWCTest
{
namespace ocam2KCtrlTest
{

/// \cond OCAM2KCTRL_TEST_SUPPORT

/// Construct a complete normal-mode table with the camera's eight interleaved amplifier lanes.
inline std::vector<int> ocamDescrambleTable();

inline std::vector<int> ocamDescrambleTable()
{
    std::vector<int> table( 240 * 240 );

    for( int row = 0; row < 240; ++row )
    {
        const int rawRow = row < 120 ? row + 1 : 240 - row;

        for( int column = 0; column < 240; ++column )
        {
            const int amplifier       = column / 60;
            const int pixel           = column % 60;
            const int lane            = row < 120 ? amplifier : 7 - amplifier;
            const int rawColumn       = amplifier % 2 == 0 ? 65 - pixel : 6 + pixel;
            table[row * 240 + column] = rawRow * 528 + rawColumn * 8 + lane;
        }
    }

    return table;
}

/// \endcond

} // namespace ocam2KCtrlTest
} // namespace libXWCTest

#endif // ocam2KCtrl_tests_ocamDescrambleFixture_hpp
