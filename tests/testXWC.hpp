/// \file testXWC.hpp
/** Support for testing of the XWCtk
 *
 */
#include "catch2/catch.hpp"

#ifndef testXWC_hpp
    #define testXWC_hpp

    /// This inserts an unused call to a function signature to make doxygen make the link
    /**  Use this if a test uses a special namespace which would prevent doxygen from auto-linking,
     * or to force reference to a protected or private member of a class.
     *
     * \param fxn is the function call of the non-special-namespace function that is being tested
     *
     * \ingroup testing
     */
    #define XWCTEST_DOXYGEN_REF( fxn )                                                                                 \
        if constexpr( false )                                                                                          \
        {                                                                                                              \
            fxn;                                                                                                       \
        }

/// Namespace for all libXWC tests
/** \ingroup tests
 *
 */
namespace libXWCTest
{
} // namespace libXWCTest

#endif
