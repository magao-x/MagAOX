/** \file defaults.hpp 
  * \brief Defaults for the MagAO-X library
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-01-05 created by JRM
  */ 

#ifndef common_defaults_hpp
#define common_defaults_hpp
   
/** \defgroup default_paths Default Paths
  * @{
  */

#ifndef MAGAOX_default_path 
   /// The default path to the MagAO-X system files
   /** Intended to include config, log, and sys directories.
     */
   #define MAGAOX_default_path "/opt/MagAOX"
#endif 

#ifndef MAGAOX_default_configRelPath
   /// The defautl relative path to the configuration files.
   /**
     */
   #define MAGAOX_default_configRelPath "config"
#endif

#ifndef MAGAOX_default_globalConfig
   /// The default filename for the global configuration file.
   /** Will be looked for the config/ directory.
     */
   #define MAGAOX_default_globalConfig "magaox.conf"
#endif

#ifndef MAGAOX_default_logRelPath
   /// The default relative path to the log directory.
   /**
     */
   #define MAGAOX_default_logRelPath "logs"
#endif

#ifndef MAGAOX_default_sysRelPath
   /// The default relative path to the system directory
   /**
     */
   #define MAGAOX_default_sysRelPath "sys"
#endif

#ifndef MAGAOX_default_secretsRelPath
   /// The default relative path to the secrets directory. Use for storing passwords, etc.
   /**
     */
   #define MAGAOX_default_secretsRelPath "secrets"
#endif

///@}

/** \defgroup default_app Default App Setup 
  * @{
  */

#ifndef MAGAOX_default_logExt
   /// The extension for MagAO-X binary log files.  
   /** Do not include period before name here.
     */
   #define MAGAOX_default_logExt "binlog" 
#endif

#ifndef MAGAOX_default_writePause
   /// The default logger writePause
   /** Defines how long the logger write thread pauses after clearing the queue.  Default is 1 sec.
     * 
     * Units: nanoseconds.
     */
   #define MAGAOX_default_writePause (1000000000)
#endif

#ifndef MAGAOX_default_logSize
   /// The default maximum log file size 
   /** Defines the maximum size in for a log file.  Default is 10 MB.
     * 
     * Units: bytes
     */
   #define MAGAOX_default_max_logSize (10485760)
#endif

#ifndef MAGAOX_default_loopPause
   /// The default application loopPause
   /** Defines how long the event loop in execute() pauses. Default is 1 sec.
     * 
     * Units: nanoseconds.
     */
   #define MAGAOX_default_loopPause (1000000000)
#endif

///@}

#endif //common_defaults_hpp
