/** \file zaberStage.hpp
  * \brief A class with details of a single zaber stage
  *
  * \ingroup zaberLowLevel_files
  */

#ifndef zaberStage_hpp
#define zaberStage_hpp

#include <iostream>


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch

#include "za_serial.h"


namespace MagAOX
{
namespace app
{

/// A class to manage the details of one stage in a Zaber system.
/**
  * \ingroup zaberLowLevel 
  */  
class zaberStage
{
protected:
   std::string m_name; ///< The stage's name.
   
   std::string m_serial; ///< The stage's serial number.
   
   int m_deviceAddress; ///< The device's address, a.k.a. its order in the chain

   int m_axisNumber{0}; ///< The axis number at the address (normally 0 in MagAO-X)

   bool m_commandStatus {true}; ///< The status of the last command sent. true = OK, false = RJ (rejected)

   char m_deviceStatus {'U'}; ///< Current status.  Either 'I' for IDLE or 'B' for BUSY.  Intializes to 'U' for UNKOWN.

   long m_rawPos; ///< The raw position reported by the device, in microsteps.
   
   long m_maxPos; ///< The max position allowed for the device, set by config.  Will be set to no larger m_maxPosHW.
   
   long m_maxPosHW; ///< The max position allowed for the device, as reported by the device.

   bool m_warn {false};
   
   bool m_warnFD {false};
   bool m_warnFDreported {false};
   bool m_warnFQ {false};
   bool m_warnFQreported {false};
   bool m_warnFS {false};
   bool m_warnFSreported {false};
   bool m_warnFT {false};
   bool m_warnFTreported {false};
   bool m_warnFB {false};
   bool m_warnFBreported {false};
   bool m_warnFP {false};
   bool m_warnFPreported {false};
   bool m_warnFE {false};
   bool m_warnFEreported {false};
   bool m_warnWH {false};
   bool m_warnWHreported {false};
   bool m_warnWL {false};
   bool m_warnWLreported {false};
   bool m_warnWP {false};
   bool m_warnWPreported {false};
   bool m_warnWV {false};
   bool m_warnWVreported {false};
   bool m_warnWT {false};
   bool m_warnWTreported {false};
   bool m_warnWM {false};
   bool m_warnWMreported {false};
   bool m_warnWR {false};
   bool m_warnWRreported {false};
   bool m_warnNC {false};
   bool m_warnNCreported {false};
   bool m_warnNI {false};
   bool m_warnNIreported {false};
   bool m_warnND {false};
   bool m_warnNDreported {false};
   bool m_warnNU {false};
   bool m_warnNUreported {false};
   bool m_warnNJ {false};
   bool m_warnNJreported {false};
   bool m_warnUNK {false};
   
public:

   /// Get the device name
   /**
     * \returns the current value of m_name
     */
   std::string name();

   /// Set the device name
   /**
     * \returns 0 on success 
     * \returns -1 on error 
     */ 
   int name( const std::string & n /**< [in] the new device name*/);
   
   /// Get the device serial  number
   /**
     * \returns the current value of m_serial
     */
   std::string serial();

   /// Set the device serial
   /**
     * \returns 0 on success 
     * \returns -1 on error 
     */ 
   int serial( const std::string & s /**< [in] the new device serial*/);
   
   /// Get the device address
   /**
     * \returns the current value of m_deviceAddress
     */
   int deviceAddress();

   /// Set the device address
   /**
     * \returns 0 on success 
     * \returns -1 on error 
     */ 
   int deviceAddress( const int & da /**< [in] the new device address*/);

   /// Get the axis number
   /**
     * \returns the current value of m_axisNumber
     */
   int axisNumber();

   /// Set the axis number
   /**
     * \returns 0 on success 
     * \returns -1 on error 
     */
   int axisNumber( const int & an /**< [in] the new axis number */);

   /// Get the command status
   /**
     * \returns the current value of m_commandStatus
     */
   bool commandStatus();

   /// Get the device status
   /**
     * \returns the current value of m_deviceStatus
     */
   char deviceStatus();

   /// Get the current raw position, in counts
   /**
     * \returns the current value of m_rawPos
     */
   long rawPos();
   
   /// Get the warning state
   /**
     * \returns the true if any warning flags are set.
     */
   bool warningState();

   bool warnFD();
   bool warnFQ();
   bool warnFS();
   bool warnFT();
   bool warnFB();
   bool warnFP();
   bool warnFE();
   bool warnWH();
   bool warnWL();
   bool warnWP();
   bool warnWV();
   bool warnWT();
   bool warnWM();
   bool warnWR();
   bool warnNC();
   bool warnNI();
   bool warnND();
   bool warnNU();
   bool warnNJ();
   bool warnUNK();
   
   /// Get a response from the device, after a command has been sent.
   /** Parses the standard parts of the response in this stage's fields,
     * and extracts the response string.
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int getResponse( std::string & response, ///< [out]  the text response
                    const std::string & repBuff ///< [in] the reply buffer, not decoded.
                  );

   /// Get a response from the device, after a command has been sent.
   /** Parses the standard parts of the response in this stages fields,
     * and extracts the response string.
     * 
     * \overload
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int getResponse( std::string & response,  ///< [out] the text response component
                    const za_reply & rep     ///< [in] the decodedstage reply
                  );

   int sendCommand( std::string & response,  ///< [out] the response received from the stage
                    z_port port,   ///< [in]  the port with which to communicate 
                    const std::string & command  ///< [in] the command to send
                   );

   int updatePos( z_port port /**< [in] the port with which to communicate */ );
   
   int stop (z_port port );
   
   int estop (z_port port );
   
   int home( z_port port /**< [in] the port with which to communicate */ );
   
   int moveAbs( z_port port, ///< [in] the port with which to communicate 
                long rawPos ///< [in] the position to move to, in counts 
              );
   
   /// Sets all warning flags to false
   /** This is not the same as clearing warnings on the device, this is just used for 
     * bookkeeping.
     * 
     * \returns 0 on success (always)
     */
   int unsetWarnings();
   
   /// Process a single warning from the device, setting the appropriate flag.
   /** Warnings are two ASCII characeters, e.g. "WR".
     * 
     * \returns 0 if the warning is processed, including if it's not recognized.
     * \returns -1 on an error, currently not possible.
     */ 
   int processWarning( std::string & warn /**< [in] the two-character warning flag */);
   
   /// Parse the warning response from the device.
   /** Sends each warning flag to processWarning.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
   int parseWarnings(std::string & response /**< [in] the response from the warnings query*/);
   
   /// Get warnings from the device
   /** Log entries will be made and flags will be set in this structure.
     * 
     * \returns 0 on success
     * \returns -1 on error.
     */ 
   int getWarnings( z_port port /**< [in] the port with which to communicate */ );

};

std::string zaberStage::name()
{
   return m_name;
}

int zaberStage::name( const std::string & n )
{
   m_name = n;
   return 0;
}

std::string zaberStage::serial()
{
   return m_serial;
}

int zaberStage::serial( const std::string & s )
{
   m_serial = s;
   return 0;
}

int zaberStage::deviceAddress()
{
   return m_deviceAddress;
}

int zaberStage::deviceAddress( const int & da )
{
   m_deviceAddress = da;
   return 0;
}

int zaberStage::axisNumber()
{
   return m_axisNumber;
}

int zaberStage::axisNumber( const int & an )
{
   m_axisNumber = an;
   return 0;
}

bool zaberStage::commandStatus()
{
   return m_commandStatus;
}

char zaberStage::deviceStatus()
{
   return m_deviceStatus;
}

inline
long zaberStage::rawPos()
{
   return m_rawPos;
}

bool zaberStage::warningState()
{
   return m_warn;
}

inline
bool zaberStage::warnFD()
{
   return m_warnFD;
}

inline
bool zaberStage::warnFQ()
{
   return m_warnFQ;
}

inline
bool zaberStage::warnFS()
{
   return m_warnFS;
}

inline
bool zaberStage::warnFT()
{
   return m_warnFT;
}

inline
bool zaberStage::warnFB()
{
   return m_warnFB;
}

inline
bool zaberStage::warnFP()
{
   return m_warnFP;
}

inline
bool zaberStage::warnFE()
{
   return m_warnFE;
}

inline
bool zaberStage::warnWH()
{
   return m_warnWH;
}

inline
bool zaberStage::warnWL()
{
   return m_warnWL;
}

inline
bool zaberStage::warnWP()
{
   return m_warnWP;
}

inline
bool zaberStage::warnWV()
{
   return m_warnWV;
}

inline
bool zaberStage::warnWT()
{
   return m_warnWT;
}

inline 
bool zaberStage::warnWM()
{
   return m_warnWM;
}

inline
bool zaberStage::warnWR()
{
   return m_warnWR;
}

inline
bool zaberStage::warnNC()
{
   return m_warnNC;
}

inline
bool zaberStage::warnNI()
{
   return m_warnNI;
}

inline 
bool zaberStage::warnND()
{
   return m_warnND;
}

inline
bool zaberStage::warnNU()
{
   return m_warnNU;
}

inline
bool zaberStage::warnNJ()
{
   return m_warnNJ;
}

inline
bool zaberStage::warnUNK()
{
   return m_warnUNK;
}

inline
int zaberStage::getResponse( std::string & response, 
                             const std::string & repBuff
                           )
{
   za_reply rep;
   int rv = za_decode(&rep, repBuff.c_str());
   if(rv != Z_SUCCESS)
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__, rv, "za_decode !=Z_SUCCESS"});      
      return rv;
   }

   return getResponse(response, rep);
}

inline
int zaberStage::getResponse( std::string & response, 
                             const za_reply & rep
                           )
{
   if(rep.device_address == m_deviceAddress)
   {
      if(rep.reply_flags[0] == 'O') m_commandStatus = true;
      else m_commandStatus = false;

      m_deviceStatus = rep.device_status[0];

      if(rep.warning_flags[0] == '-') unsetWarnings();
      else m_warn = true;;

      response = rep.response_data;

      return 0;
   }
   else 
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__, 0, "wrong device"});
      return -1;
   }
}
 
inline
int zaberStage::sendCommand( std::string & response, 
                             z_port port, 
                             const std::string & command
                           )
{
   MagAOXAppT::log<text_log>(std::string("Sending: ") + command, logPrio::LOG_DEBUG2);
   
   za_send(port, command.c_str());

   char buff[256];

   while(1)
   {
      int rv = za_receive(port, buff, sizeof(buff));
      if(rv == Z_ERROR_TIMEOUT)
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, 0, "Z_ERROR_TIMEOUT"});
         break; //assume error and just get out.
      }
      else if(rv < 0)
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, 0, "za_receive !=Z_SUCCESS"});
         break;
      }
      za_reply rep;

      
      MagAOXAppT::log<text_log>(std::string("Received: ") + buff, logPrio::LOG_DEBUG2);

      rv = za_decode(&rep, buff);
      if(rv != Z_SUCCESS)
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, 0, "za_decode !=Z_SUCCESS"});
         break;
      }

      if(rep.device_address == m_deviceAddress) return getResponse(response, rep);
   }

   response = "";

   return -1;
}

inline
int zaberStage::updatePos( z_port port )
{
   std::string com = "/" + mx::ioutils::convertToString(m_deviceAddress) + " ";
   com += "get pos";

   std::string response;

   int rv = sendCommand(response, port, com);

   if(rv == 0)
   {
      if( m_commandStatus )
      {
         m_rawPos = mx::ioutils::convertFromString<long>(response);
         return 0;
      }
      else
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, rv, "get pos Command Rejected"});         
         return -1;
      }
   }
   else
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__});
      return -1;
   }
}

inline
int zaberStage::stop( z_port port )
{
   std::string com = "/" + mx::ioutils::convertToString(m_deviceAddress) + " ";
   com += "stop";

   std::string response;

   int rv = sendCommand(response, port, com);

   if(rv == 0)
   {
      if( m_commandStatus )
      {
         return 0;
      }
      else
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, rv, m_name + " stop Command Rejected"});         
         return -1;
      }
   }
   else
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__});
      return -1;
   }
}

inline
int zaberStage::estop( z_port port )
{
   std::string com = "/" + mx::ioutils::convertToString(m_deviceAddress) + " ";
   com += "estop";

   std::string response;

   int rv = sendCommand(response, port, com);

   if(rv == 0)
   {
      if( m_commandStatus )
      {
         return 0;
      }
      else
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, rv, m_name + " estop Command Rejected"});         
         return -1;
      }
   }
   else
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__});
      return -1;
   }
}

inline
int zaberStage::home( z_port port )
{
   std::string com = "/" + mx::ioutils::convertToString(m_deviceAddress) + " ";
   com += "home";

   std::string response;

   int rv = sendCommand(response, port, com);

   if(rv == 0)
   {
      if( m_commandStatus )
      {
         return 0;
      }
      else
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, rv, m_name + "Home Command Rejected"});         
         return -1;
      }
   }
   else
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__});
      return -1;
   }
}

inline
int zaberStage::moveAbs( z_port port,
                         long rawPos
                       )
{
   std::string com = "/" + mx::ioutils::convertToString(m_deviceAddress) + " ";
   com += "move abs " + std::to_string(rawPos);

   std::string response;

   int rv = sendCommand(response, port, com);

   if(rv == 0)
   {
      if( m_commandStatus )
      {
         return 0;
      }
      else
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, rv, m_name + "move abs Command Rejected"});         
         return -1;
      }
   }
   else
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__});
      return -1;
   }
}

inline
int zaberStage::unsetWarnings()
{
   m_warn = false;
   
   m_warnFD = false;
   m_warnFQ = false;
   m_warnFS = false;
   m_warnFT = false;
   m_warnFB = false;
   m_warnFP = false;
   m_warnFE = false;
   m_warnWH = false;
   m_warnWL = false;
   m_warnWP = false;
   m_warnWV = false;
   m_warnWT = false;
   m_warnWM = false;
   m_warnWR = false;
   m_warnNC = false;
   m_warnNI = false;
   m_warnND = false;
   m_warnNU = false;
   m_warnNJ = false;
   m_warnUNK = false;
   
   return 0;
}

inline
int zaberStage::processWarning( std::string & warn )
{
   if(warn == "FD")
   {
      MagAOXAppT::log<text_log>(m_name + " Driver Disabled (FD): The driver has disabled itself due to overheating." , logPrio::LOG_EMERGENCY);
      m_warnFD = true;
      return 0;
   }
   else if(warn == "FQ")
   {
      MagAOXAppT::log<text_log>(m_name + " Encoder Error (FQ): The encoder-measured position may be unreliable. [home recommended]" , logPrio::LOG_WARNING);
      m_warnFQ = true;
      return 0;
   }
   else if(warn == "FS")
   {
      MagAOXAppT::log<text_log>(m_name + " Stalled and Stopped (FS): Stalling was detected and the axis has stopped itself. " , logPrio::LOG_WARNING);
      m_warnFS = true;
      return 0;
   }
   else if(warn == "FT")
   {
      MagAOXAppT::log<text_log>(m_name + " Excessive Twist (FT): The lockstep group has exceeded allowable twist and has stopped. " , logPrio::LOG_WARNING);
      m_warnFT = true;
      return 0;
   }
   else if(warn == "FB")
   {
      MagAOXAppT::log<text_log>(m_name + " Stream Bounds Error (FB): A previous streamed motion could not be executed because it failed a precondition" , logPrio::LOG_WARNING);
      m_warnFB = true;
      return 0;
   }
   else if(warn == "FP")
   {
      MagAOXAppT::log<text_log>(m_name + " Interpolated Path Deviation (FP): Streamed or sinusoidal motion was terminated because an axis slipped and thus the device deviated from the requested path. " , logPrio::LOG_WARNING);
      m_warnFP = true;
      return 0;
   }
   else if(warn == "FE")
   {
      MagAOXAppT::log<text_log>(m_name + " Limit Error (FE): The target limit sensor cannot be reached or is faulty. " , logPrio::LOG_WARNING);
      m_warnFE = true;
      return 0;
   }
   else if(warn == "WH")
   {
      if(m_warnWHreported == false)
      {
         MagAOXAppT::log<text_log>(m_name + " Device not homed (WH): The device has a position reference, but has not been homed." , logPrio::LOG_WARNING);
         m_warnWHreported = true;
      }
      m_warnWH = true;
      return 0;
   }
   else if(warn == "WL")
   {
      MagAOXAppT::log<text_log>(m_name + " Unexpected Limit Trigger warning (WL): A movement operation did not complete due to a triggered limit sensor." , logPrio::LOG_WARNING);
      m_warnWL = true;
      return 0;
   }
   else if(warn == "WP")
   {
      MagAOXAppT::log<text_log>(m_name + " Invalid calibration type (WP): The saved calibration data type is unsupported" , logPrio::LOG_WARNING);
      m_warnWP = true;
      return 0;
   }
   else if(warn == "WV")
   {
      MagAOXAppT::log<text_log>(m_name + " Voltage Out of Range (WV): The supply voltage is outside the recommended operating range of the device" , logPrio::LOG_WARNING);
      m_warnWV = true;
      return 0;
   }
   else if(warn == "WT")
   {
      MagAOXAppT::log<text_log>(m_name + " Controller Temperature High (WT): The internal temperature of the controller has exceeded the recommended limit for the device." , logPrio::LOG_WARNING);
      m_warnWT = true;
      return 0;
   }
   else if(warn == "WM")
   {
      MagAOXAppT::log<text_log>(m_name + " Displaced when Stationary (WM): While not in motion, the axis has been forced out of its position." , logPrio::LOG_WARNING);
      m_warnWM = true;
      return 0;
   }
   else if(warn == "WR")
   {
      if(m_warnWRreported == false)
      {
         MagAOXAppT::log<text_log>(m_name + " No Reference Position (WR): Axis has not had a reference position established. [homing required]" , logPrio::LOG_WARNING);
         m_warnWRreported = true;
      }
      m_warnWR = true;
      return 0;
   }
   else if(warn == "NC")
   {
      MagAOXAppT::log<text_log>(m_name + " Manual Control (NC): Axis is busy due to manual control via the knob." , logPrio::LOG_WARNING);
      m_warnNC = true;
      return 0;
   }
   else if(warn == "NI")
   {
      MagAOXAppT::log<text_log>(m_name + " Command Interrupted (NI): A movement operation (command or manual control) was requested while the axis was executing another movement command." , logPrio::LOG_WARNING);
      m_warnNI = true;
      return 0;
   }
   else if(warn == "ND")
   {
      MagAOXAppT::log<text_log>(m_name + " Stream Discontinuity (ND): The device has slowed down while following a streamed motion path because it has run out of queued motions." , logPrio::LOG_WARNING);
      m_warnND = true;
      return 0;
   }
   else if(warn == "NU")
   {
      MagAOXAppT::log<text_log>(m_name + " Setting Update Pending (NU): A setting is pending to be updated or a reset is pending." , logPrio::LOG_WARNING);
      m_warnNU = true;
      return 0;
   }
   else if(warn == "NJ")
   {
      MagAOXAppT::log<text_log>(m_name + " Joystick Calibrating (NJ): Joystick calibration is in progress." , logPrio::LOG_WARNING);
      m_warnNJ = true;
      return 0;
   }
   else
   {
      m_warnUNK = true;
      MagAOXAppT::log<software_warning>({__FILE__, __LINE__, m_name + " unknown stage warning: " + warn});
      return 0;
   }
   
   return -1;
}

inline
int zaberStage::parseWarnings( std::string & response )
{
   size_t nwarn = std::stoi( response.substr(0, 2));
      
   if(nwarn > 0) m_warn = true;
   
   for(size_t n =0; n< nwarn; ++n)
   {
      if(response.size() < 3 + n*3)
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, "parsing incomplete warning response"});         
         return -1;
      }
      
      std::string warn = response.substr(3 + n*3, 2);
      
      int rv = processWarning(warn);
      if(rv < 0)
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__});         
         return -1;
      }
   }
   
   
   if(m_warnWHreported)
   {
      if(!m_warnWH) m_warnWHreported = false;
   }
   
   if(m_warnWRreported)
   {
      if(!m_warnWR) 
      {
         MagAOXAppT::log<text_log>(m_name + " homed.  WR clear." , logPrio::LOG_NOTICE);
         m_warnWRreported = false;
      }
   }
   
   return 0;
   
}

inline
int zaberStage::getWarnings( z_port port )
{
   std::string com = "/" + mx::ioutils::convertToString(m_deviceAddress) + " ";
   com += "warnings";

   std::string response;

   int rv = sendCommand(response, port, com);

   if(rv == 0)
   {
      if( m_commandStatus )
      {
         unsetWarnings(); //Clear all the flags before setting them to stay current.
         return parseWarnings(response);
         
      }
      else
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, rv, "warnings Command Rejected"});         
         return -1;
      }
   }
   else
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__});
      return -1;
   }
}


} //namespace app
} //namespace MagAOX

#endif //zaberStage_hpp
