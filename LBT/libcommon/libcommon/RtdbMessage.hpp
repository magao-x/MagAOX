/// RtdbMessage.hpp
///
/// @author Paul Grenz
/// This is the class that is used to send messages between the msgd
/// and its client.

////////////////////////////////////////////////////////////////////////////////

#ifndef MSGD_RTDB_MESSAGE_HPP
#define MSGD_RTDB_MESSAGE_HPP

#include <vector>
#include <string>
#include "RtdbVariable.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace msgd
{
class RtdbMessage
{
  public:
    enum Constants
    {
      StdClient =     1,
      PeerMsgD =      2,
      Protocol =      44,      // 'Magic' protocol number for msgd.
      LongNameLen =   40,
    };

  public:
    enum Type
    {
      AnyMessage =    0x0,     // Any message code
      Ack =           0x10,    // Generic acknowledge message
      Nak =           0x20,    // Generic negative acknowledge message
      Register =      0x1010,  // Registers a client. Replies with a ACK/NAK message
      SetReady =      0x1020,  // Sets the ready flag of a client
      SetClientRdy  = 0x1030,  // Sets the client waiting for given client to become ready.
      WaitCancel =    0x1031,  // Cancel waiting for client ready
      GetVersion =    0x1040,  // Request numeric version (1000*VERS_MAJOR+VERS_MINOR)
      VersReply =     0x1040,  // Reply to version request
      GetStartTime =  0x1044,  // Request client's start time
      StartReply =    0x1045,  // Reply to stat time request
      Unregister =    0x1100,  // Unregister client (client will then close the connection)

      CreateVar =     0x5000,  // Creates a variable
      SetVar =        0x5010,  // Sets a Variable
      GetVar =        0x5020,  // Gets a Variable
      DeleteVar =     0x5030,  // Deletes a Variable
      LockVar =       0x5040,  // Lock/unlock a set of variables (spec: 1/0 in payload)
      TraceVar =      0x5042,  // Set/unset trace flag on var (spec: 1/0 in payload)
      WriteVar =      0x5050,  // Writes a variable (creates it, if not existent)
      VarList =       0x5070,  // Request a list of variable values
      VarNotify =     0x5080,  // Request/unrequest var change notification (spec: 1/0 in payload)
      TakeVar =       0x5100,  // Take ownership of a variable
      GiveVar =       0x5110,  // Give away ownership of a variable
      VarChanged =    0x5500,  // Notify variable change to registered clients (details in payload)
      VarReply =      0x5530,  // Variable value return
      VarEnd =        0x5540,  // Terminates a sequence of VARREPLY
      VarError =      0x5550,  // Variable error return
      VarLocked =     0x5560,  // Variable locked error return
      VarWritten =    0x5570,  // Variable has been written (reply to WRITEVAR)
    };

   enum FlagType
   {
     NoFlag =         0x0,    // No flag specified.
     NoHandle =       0x01,   // Reply message will be forcibily put into the message queue.
     Discard =        0x02,   // Reply message will be discarded.
     Reply =          0x40,   // Handle the reply.
   };

  public:
    #pragma pack( push, 1 )
    struct Signature
    {
      uint32_t uiMagic;
      uint32_t uiLen;
    };
    struct Header
    {
      uint32_t uiSeqNum;
      uint32_t uiCode;
      uint32_t uiFlags;
      int32_t iTtl;
      int32_t iPayload;
      char pcFrom[LongNameLen];
      char pcTo[LongNameLen];
    };
    #pragma pack( pop ) // Back to whatever the previous packing mode was.

  public:
    RtdbMessage();
    virtual ~RtdbMessage();
    RtdbMessage( const RtdbMessage &msg );
    // Assume that the size in the signature is the correct number of bytes.
    RtdbMessage( const Signature &sig,
                 const Header &hdr,
                 const std::vector<char> &vecBody );
    RtdbMessage( const uint32_t &uiMagic,
                 const uint32_t &uiLen,
                 const uint32_t &uiSeqNum,
                 const uint32_t &uiCode,
                 const uint32_t &uiFlags,
                 const int32_t &iTtl,
                 const int32_t &iPayload,
                 const std::string &szFrom,
                 const std::string &szTo,
                 const std::vector<char> &vecBody );
    RtdbMessage( const uint32_t &uiMagic,
                 const uint32_t &uiSeqNum,
                 const uint32_t &uiCode,
                 const uint32_t &uiFlags,
                 const int32_t &iTtl,
                 const int32_t &iPayload,
                 const std::string &szFrom,
                 const std::string &szTo,
                 const std::string &szBody );
    RtdbMessage( const uint32_t &uiMagic,
                 const uint32_t &uiSeqNum,
                 const uint32_t &uiCode,
                 const uint32_t &uiFlags,
                 const int32_t &iTtl,
                 const int32_t &iPayload,
                 const std::string &szFrom,
                 const std::string &szTo,
                 const msgd::RtdbVariable &varMessage );
    RtdbMessage( const uint32_t &uiMagic,
                 const uint32_t &uiSeqNum,
                 const uint32_t &uiCode,
                 const uint32_t &uiFlags,
                 const int32_t &iTtl,
                 const int32_t &iPayload,
                 const std::string &szFrom,
                 const std::string &szTo );

  public:
    const RtdbMessage &operator=( const RtdbMessage &var );

  public:
    const std::vector<char> &getBody() const;
    std::string getBodyString() const;
    const msgd::RtdbMessage::Header &getHeader() const;
    const msgd::RtdbMessage::Signature &getSignature() const;
    msgd::RtdbVariable getVariable() const;

  private:
    Signature m_sigMsg;
    Header m_hdrMsg;
    std::vector<char> m_vecBody;

}; // class RtdbMessage
} // namespace msgd

#endif // MSGD_RTDB_MESSAGE_HPP
