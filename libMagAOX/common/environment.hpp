/** \file environment.hpp 
  * \brief Environment variables for the MagAO-X library
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-01-05 created by JRM
  */ 

#ifndef common_environment_hpp
#define common_environment_hpp

/** \defgroup env_var Environment Variables
  * \ingroup common 
  * 
  * @{
  */

#ifndef MAGAOX_env_path
   /// Environment variable setting the MagAO-X path.
   #define MAGAOX_env_path "MagAOX_PATH"
#endif 

#ifndef MAGAOX_env_config
   /// Environment variable setting the relative config path.
   #define MAGAOX_env_config "MagAOX_CONFIG"
#endif

#ifndef MAGAOX_env_calib
   /// Environment variable setting the relative calib path.
   #define MAGAOX_env_calib "MagAOX_CALIB"
#endif

#ifndef MAGAOX_env_cpuset
   /// Environment variable setting the relative calib path.
   #define MAGAOX_env_cpuset "CGROUPS1_CPUSET_MOUNTPOINT"
#endif


///@}


#endif //common_environment_hpp
