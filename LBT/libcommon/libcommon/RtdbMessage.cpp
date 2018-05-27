/// RtdbMessage.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <sstream>
#include "RtdbMessage.hpp"

using std::string;
using std::stringstream;
using std::vector;
using pcf::TimeStamp;
using msgd::RtdbMessage;
using msgd::RtdbVariable;

////////////////////////////////////////////////////////////////////////////////

RtdbMessage::RtdbMessage()
{
  m_sigMsg.uiMagic = 0;
  m_sigMsg.uiLen = 0;
  m_hdrMsg.uiSeqNum = 0;
  m_hdrMsg.uiCode = 0;
  m_hdrMsg.uiFlags = 0;
  m_hdrMsg.iTtl = 0;
  m_hdrMsg.iPayload = 0;
  ::memset( m_hdrMsg.pcFrom, 0, LongNameLen );
  ::memset( m_hdrMsg.pcTo, 0, LongNameLen );
}

////////////////////////////////////////////////////////////////////////////////

RtdbMessage::~RtdbMessage()
{
}

////////////////////////////////////////////////////////////////////////////////

RtdbMessage::RtdbMessage( const RtdbMessage &msg )
{
  m_sigMsg.uiMagic = msg.m_sigMsg.uiMagic;
  m_sigMsg.uiLen = msg.m_sigMsg.uiLen;
  m_hdrMsg.uiSeqNum = msg.m_hdrMsg.uiSeqNum;
  m_hdrMsg.uiCode = msg.m_hdrMsg.uiCode;
  m_hdrMsg.uiFlags = msg.m_hdrMsg.uiFlags;
  m_hdrMsg.iTtl = msg.m_hdrMsg.iTtl;
  m_hdrMsg.iPayload = msg.m_hdrMsg.iPayload;
  ::memset( m_hdrMsg.pcFrom, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcFrom, msg.m_hdrMsg.pcFrom, LongNameLen );
  ::memset( m_hdrMsg.pcTo, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcTo, msg.m_hdrMsg.pcTo, LongNameLen );
  m_vecBody = msg.m_vecBody;
}

////////////////////////////////////////////////////////////////////////////////

const RtdbMessage &RtdbMessage::operator=( const RtdbMessage &msg )
{
  if ( this != &msg )
  {
    m_sigMsg.uiMagic = msg.m_sigMsg.uiMagic;
    m_sigMsg.uiLen = msg.m_sigMsg.uiLen;
    m_hdrMsg.uiSeqNum = msg.m_hdrMsg.uiSeqNum;
    m_hdrMsg.uiCode = msg.m_hdrMsg.uiCode;
    m_hdrMsg.uiFlags = msg.m_hdrMsg.uiFlags;
    m_hdrMsg.iTtl = msg.m_hdrMsg.iTtl;
    m_hdrMsg.iPayload = msg.m_hdrMsg.iPayload;
    ::memset( m_hdrMsg.pcFrom, 0, LongNameLen );
    ::memcpy( m_hdrMsg.pcFrom, msg.m_hdrMsg.pcFrom, LongNameLen );
    ::memset( m_hdrMsg.pcTo, 0, LongNameLen );
    ::memcpy( m_hdrMsg.pcTo, msg.m_hdrMsg.pcTo, LongNameLen );
    m_vecBody = msg.m_vecBody;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
// Assume that the size in the signature is the correct number of bytes.

RtdbMessage::RtdbMessage( const Signature &sig,
                          const Header &hdr,
                          const std::vector<char> &vecBody )
{
  m_sigMsg.uiMagic = sig.uiMagic;
  m_sigMsg.uiLen = sig.uiLen;
  m_hdrMsg.uiSeqNum = hdr.uiSeqNum;
  m_hdrMsg.uiCode = hdr.uiCode;
  m_hdrMsg.uiFlags = hdr.uiFlags;
  m_hdrMsg.iTtl = hdr.iTtl;
  m_hdrMsg.iPayload = hdr.iPayload;
  ::memset( m_hdrMsg.pcFrom, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcFrom, hdr.pcFrom, LongNameLen );
  ::memset( m_hdrMsg.pcTo, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcTo, hdr.pcTo, LongNameLen );
  m_vecBody = vecBody;
}

////////////////////////////////////////////////////////////////////////////////

RtdbMessage::RtdbMessage( const uint32_t &uiMagic,
                          const uint32_t &uiLen,
                          const uint32_t &uiSeqNum,
                          const uint32_t &uiCode,
                          const uint32_t &uiFlags,
                          const int32_t &iTtl,
                          const int32_t &iPayload,
                          const std::string &szFrom,
                          const std::string &szTo,
                          const std::vector<char> &vecBody )
{
  m_sigMsg.uiMagic = uiMagic;
  m_sigMsg.uiLen = uiLen;
  m_hdrMsg.uiSeqNum = uiSeqNum;
  m_hdrMsg.uiCode = uiCode;
  m_hdrMsg.uiFlags = uiFlags;
  m_hdrMsg.iTtl = iTtl;
  m_hdrMsg.iPayload = iPayload;
  ::memset( m_hdrMsg.pcFrom, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcFrom, szFrom.c_str(), szFrom.size() );
  ::memset( m_hdrMsg.pcTo, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcTo, szTo.c_str(), szTo.size() );
  m_vecBody = vecBody;
}

////////////////////////////////////////////////////////////////////////////////

RtdbMessage::RtdbMessage(const uint32_t &uiMagic,
                          const uint32_t &uiSeqNum,
                          const uint32_t &uiCode,
                          const uint32_t &uiFlags,
                          const int32_t &iTtl,
                          const int32_t &iPayload,
                          const std::string &szFrom,
                          const std::string &szTo,
                          const std::string &szBody )
{
  m_sigMsg.uiMagic = uiMagic;
  m_sigMsg.uiLen = szBody.size() + 1;
  m_hdrMsg.uiSeqNum = uiSeqNum;
  m_hdrMsg.uiCode = uiCode;
  m_hdrMsg.uiFlags = uiFlags;
  m_hdrMsg.iTtl = iTtl;
  m_hdrMsg.iPayload = iPayload;
  ::memset( m_hdrMsg.pcFrom, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcFrom, szFrom.c_str(), szFrom.size() );
  ::memset( m_hdrMsg.pcTo, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcTo, szTo.c_str(), szTo.size() );

  // The "+1" is to ensure we have a "\0" at the end.
  m_vecBody = vector<char>( szBody.size() + 1 );
  ::memset( &m_vecBody[0], 0, szBody.size() + 1 );
  ::memcpy( &m_vecBody[0], szBody.c_str(), szBody.size() );
}

////////////////////////////////////////////////////////////////////////////////

RtdbMessage::RtdbMessage(const uint32_t &uiMagic,
                          const uint32_t &uiSeqNum,
                          const uint32_t &uiCode,
                          const uint32_t &uiFlags,
                          const int32_t &iTtl,
                          const int32_t &iPayload,
                          const std::string &szFrom,
                          const std::string &szTo,
                          const RtdbVariable &varMessage )
{
  unsigned int uiHeaderSize = sizeof( RtdbVariable::Header );
  unsigned int uiNumBytes = varMessage.getNumBytes();

  m_sigMsg.uiMagic = uiMagic;
  m_sigMsg.uiLen = uiHeaderSize + uiNumBytes;
  m_hdrMsg.uiSeqNum = uiSeqNum;
  m_hdrMsg.uiCode = uiCode;
  m_hdrMsg.uiFlags = uiFlags;
  m_hdrMsg.iTtl = iTtl;
  m_hdrMsg.iPayload = iPayload;
  ::memset( m_hdrMsg.pcFrom, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcFrom, szFrom.c_str(), szFrom.size() );
  ::memset( m_hdrMsg.pcTo, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcTo, szTo.c_str(), szTo.size() );

  // Get a pointer to the characters contained in the vector.
  const vector<char> &vecValues = varMessage.getValues();
  const RtdbVariable::Header &hdrVar = varMessage.getHeader();
  const char *pch = &vecValues[0];
  const char *phd = ( const char* )( &hdrVar );

  // Now copy the data from the variable to the body.
  m_vecBody = vector<char>( uiHeaderSize + uiNumBytes );
  ::memset( &m_vecBody[0], 0, uiHeaderSize + uiNumBytes );
  ::memcpy( &m_vecBody[0], phd, uiHeaderSize );
  ::memcpy( &m_vecBody[uiHeaderSize], pch, uiNumBytes );
}

////////////////////////////////////////////////////////////////////////////////

RtdbMessage::RtdbMessage(const uint32_t &uiMagic,
                          const uint32_t &uiSeqNum,
                          const uint32_t &uiCode,
                          const uint32_t &uiFlags,
                          const int32_t &iTtl,
                          const int32_t &iPayload,
                          const std::string &szFrom,
                          const std::string &szTo )
{
  m_sigMsg.uiMagic = uiMagic;
  m_sigMsg.uiLen = 0;
  m_hdrMsg.uiSeqNum = uiSeqNum;
  m_hdrMsg.uiCode = uiCode;
  m_hdrMsg.uiFlags = uiFlags;
  m_hdrMsg.iTtl = iTtl;
  m_hdrMsg.iPayload = iPayload;
  ::memset( m_hdrMsg.pcFrom, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcFrom, szFrom.c_str(), szFrom.size() );
  ::memset( m_hdrMsg.pcTo, 0, LongNameLen );
  ::memcpy( m_hdrMsg.pcTo, szTo.c_str(), szTo.size() );

  m_vecBody.clear();
}

////////////////////////////////////////////////////////////////////////////////
// Return the variable.

RtdbVariable RtdbMessage::getVariable() const
{
  RtdbVariable::Header *phdr = (RtdbVariable::Header*)( &m_vecBody[0] );
  char *pch = (char*)( &m_vecBody[sizeof(RtdbVariable::Header)] );
  int iNumBytes = m_vecBody.size() - sizeof( RtdbVariable::Header );
  return RtdbVariable( *phdr, std::vector<char>( pch, pch + iNumBytes ) );
}

////////////////////////////////////////////////////////////////////////////////

const vector<char> &RtdbMessage::getBody() const
{
  return m_vecBody;
}

////////////////////////////////////////////////////////////////////////////////

string RtdbMessage::getBodyString() const
{
  return string( m_vecBody.begin(), m_vecBody.end() );
}

////////////////////////////////////////////////////////////////////////////////

const RtdbMessage::Header &RtdbMessage::getHeader() const
{
  return m_hdrMsg;
}

////////////////////////////////////////////////////////////////////////////////

const RtdbMessage::Signature &RtdbMessage::getSignature() const
{
  return m_sigMsg;
}

////////////////////////////////////////////////////////////////////////////////
