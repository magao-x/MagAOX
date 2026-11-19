/** \file operational_platform.cpp
 * \brief macOS link double for native-header tests whose configuration prohibits output; not a Linux timing test.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include <cerrno>
#include <time.h>

#if defined( __APPLE__ )
/// Refuse this absent Linux operation; actual producer transport tests inject their own deterministic clock.
int clock_nanosleep( clockid_t       clock /**< [in] Unused clock. */,
                     int             flags /**< [in] Unused wait mode. */,
                     const timespec *requested /**< [in] Unused deadline. */,
                     timespec       *remaining /**< [out] Unused remainder. */ );

int clock_nanosleep( clockid_t, int, const timespec *, timespec * )
{
    return ENOTSUP;
}
#endif
