/** \file paths.hpp
  * \brief Defaults for the MagAO-X library
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-07-13 created by JRM
  */

#ifndef common_paths_hpp
#define common_paths_hpp

/** \defgroup default_paths Default Paths
  * \ingroup common
  *
  * @{
  */

#ifndef MAGAOX_path
   /// The path to the MagAO-X system files
   /** This directory will have subdirectories including config, log, and sys directories.
     */
   #define MAGAOX_path "/opt/MagAOX"
#endif

#ifndef MAGAOX_configRelPath
   /// The relative path to the configuration files.
   /** This is the subdirectory for configuration files.
     */
   #define MAGAOX_configRelPath "config"
#endif

#ifndef MAGAOX_calibRelPath
   /// The relative path to the calibration files.
   /** This is the subdirectory for calibration files.
     */
   #define MAGAOX_calibRelPath "calib"
#endif

#ifndef MAGAOX_globalConfig
   /// The filename for the global configuration file.
   /** Will be looked for in the config/ subdirectory.
     */
   #define MAGAOX_globalConfig "magaox.conf"
#endif

#ifndef MAGAOX_logRelPath
   /// The relative path to the log directory.
   /** This is the subdirectory for logs.
     */
   #define MAGAOX_logRelPath "logs"
#endif

#ifndef MAGAOX_telRelPath
   /// The relative path to the telemetry directory.
   /** This is the subdirectory for telemetry.
     */
   #define MAGAOX_telRelPath "telem"
#endif

#ifndef MAGAOX_sysRelPath
   /// The relative path to the system directory
   /** This is the subdirectory for the system status files.
     */
   #define MAGAOX_sysRelPath "sys"
#endif

#ifndef MAGAOX_secretsRelPath
   /// The relative path to the secrets directory. Used for storing passwords, etc.
   /** This is the subdirectory for secrets.
     */
   #define MAGAOX_secretsRelPath "secrets"
#endif

#ifndef MAGAOX_driverRelPath
   /// The relative path to the INDI drivers
   /** This is the subdirectory for the INDI drivers.
     */
   #define MAGAOX_driverRelPath "drivers"
#endif

#ifndef MAGAOX_driverFIFORelPath
   /// The relative path to the INDI driver FIFOs
   /** This is the subdirectory for the INDI driver FIFOs.
     */
   #define MAGAOX_driverFIFORelPath "drivers/fifos"
#endif

#ifndef MAGAOX_rawimageRelPath
   /// The relative path to the raw images directory.
   /** This is the subdirectory for raw images.
     */
   #define MAGAOX_rawimageRelPath "rawimages"
#endif

#ifndef MAGAOX_cpusetPath
   /// The absolute path the the cpuset mount point
   /**
     */
   #define MAGAOX_cpusetPath "/opt/MagAOX/cpuset/"
#endif

///@}

#endif //common_paths_hpp
