/** \file master.h
 * \brief PVCAM SDK base-type stubs for SDK-less MagAO-X test builds.
 *
 * Mirrors the subset of the Teledyne PVCAM `master.h` used by pvcamCtrl.  Only test builds add this directory to
 * the include path, so production builds always use the installed SDK.
 *
 * \ingroup pvcamCtrl_unit_test
 */

#ifndef tests_pvcam_master_h
#define tests_pvcam_master_h

#define PV_DECL

/// PVCAM success/failure return values.
enum
{
    PV_FAIL = 0,
    PV_OK
};

typedef unsigned short     rs_bool;
typedef signed char        int8;
typedef unsigned char      uns8;
typedef short              int16;
typedef unsigned short     uns16;
typedef int                int32;
typedef unsigned int       uns32;
typedef float              flt32;
typedef double             flt64;
typedef unsigned long long ulong64;
typedef signed long long   long64;

#endif // tests_pvcam_master_h
