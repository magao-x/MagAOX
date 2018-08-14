/** \file zaberStage.hpp
  * \brief A class with details of a single zaber stage
  *
  * \ingroup zaberCtrl_files
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

///A class to manage the details of one stage in a Zaber system.
class zaberStage
{
protected:
   int m_deviceAddress; ///< The device's address, a.k.a. its order in the chain

   int m_axisNumber{0}; ///< The axis number at the address (normally 0 in MagAO-X)

   bool m_commandStatus {true}; ///< The status of the last command sent. true = OK, false = RJ (rejected)

   char m_deviceStatus {'U'}; ///< Current status.  Either 'I' for IDLE or 'B' for BUSY.  Intializes to 'U' for UNKOWN.

   bool m_warningState {false}; ///< True if a warning exists, false if no warning.

   long m_rawPos; ///< The raw position reported by the device, in microsteps.

public:

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

   /// Get the warning state
   /**
     * \returns the current value of m_warningState
     */
   bool warningState();

   /// Get a response from the device, after a command has been sent.
   /** Parses the standard parts of the response in this stages fields,
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

};

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

bool zaberStage::warningState()
{
   return m_warningState;
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

      if(rep.warning_flags[0] == '-') m_warningState = false;
      else m_warningState = true;

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
   MagAOXAppT::log<text_log>(std::string("Sending: ") + command, logLevels::DEBUG2);
   
   std::cerr << "Sending: " << command << "\n";
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

      
      MagAOXAppT::log<text_log>(std::string("Received: ") + buff, logLevels::DEBUG2);

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
         std::cerr << "m_rawPos: " << m_rawPos << "\n";
         return 0;
      }
      else
      {
         MagAOXAppT::log<software_error>({__FILE__, __LINE__, rv, "Command Rejected"});         
         return -1;
      }
   }
   else
   {
      MagAOXAppT::log<software_trace_error>({__FILE__, __LINE__});
      return -1;
   }
}

} //namespace app
} //namespace MagAOX

#endif //zaberStage_hpp
