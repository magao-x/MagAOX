/// RtdbVariable.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <sstream>
#include "RtdbVariable.hpp"

using std::string;
using std::stringstream;
using std::vector;
using pcf::TimeStamp;
using msgd::RtdbVariable;

////////////////////////////////////////////////////////////////////////////////

RtdbVariable::RtdbVariable()
{
  m_hdrVar.iDummy = 0;
  m_hdrVar.iNumItems = 0;
  m_hdrVar.tType = RtdbVariable::Unknown;
  ::memset( m_hdrVar.pcName, 0, VarNameLen );
  ::memset( m_hdrVar.pcOwner, 0, OwnerNameLen );
  m_hdrVar.tvModTime.tv_sec = 0;
  m_hdrVar.tvModTime.tv_usec = 0;
}

////////////////////////////////////////////////////////////////////////////////

RtdbVariable::~RtdbVariable()
{
}

////////////////////////////////////////////////////////////////////////////////

RtdbVariable::RtdbVariable( const RtdbVariable &var )
{
  m_hdrVar.iDummy = var.m_hdrVar.iDummy;
  m_hdrVar.iNumItems = var.m_hdrVar.iNumItems;
  m_hdrVar.tType = var.m_hdrVar.tType;
  ::memcpy( m_hdrVar.pcName, var.m_hdrVar.pcName, VarNameLen );
  ::memcpy( m_hdrVar.pcOwner, var.m_hdrVar.pcOwner, OwnerNameLen );
  m_hdrVar.tvModTime = var.m_hdrVar.tvModTime;
  m_vecValues = var.m_vecValues;
}

////////////////////////////////////////////////////////////////////////////////

const RtdbVariable &RtdbVariable::operator=( const RtdbVariable &var )
{
  if ( &var != this )
  {
    m_hdrVar.iDummy = var.m_hdrVar.iDummy;
    m_hdrVar.iNumItems = var.m_hdrVar.iNumItems;
    m_hdrVar.tType = var.m_hdrVar.tType;
    ::memcpy( m_hdrVar.pcName, var.m_hdrVar.pcName, VarNameLen );
    ::memcpy( m_hdrVar.pcOwner, var.m_hdrVar.pcOwner, OwnerNameLen );
    m_hdrVar.tvModTime = var.m_hdrVar.tvModTime;
    m_vecValues = var.m_vecValues;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////

RtdbVariable::RtdbVariable( const RtdbVariable::Header &hdr,
                            const vector<char> &vecValues )
{
  m_hdrVar.iDummy = hdr.iDummy;
  m_hdrVar.iNumItems = hdr.iNumItems;
  m_hdrVar.tType = hdr.tType;
  ::memcpy( m_hdrVar.pcName, hdr.pcName, VarNameLen );
  ::memcpy( m_hdrVar.pcOwner, hdr.pcOwner, OwnerNameLen );
  m_hdrVar.tvModTime = hdr.tvModTime;
  m_vecValues = vecValues;
}

////////////////////////////////////////////////////////////////////////////////

RtdbVariable::RtdbVariable( const string &szName,
                            const string &szOwner,
                            const Type &tType,
                            const int &iNumItems,
                            const TimeStamp &tsModTime )
{
  m_hdrVar.iDummy = 0;
  m_hdrVar.iNumItems = iNumItems;
  m_hdrVar.tType = tType;
  ::memset( m_hdrVar.pcName, 0, VarNameLen );
  ::memcpy( m_hdrVar.pcName, szName.c_str(), VarNameLen );
  ::memset( m_hdrVar.pcOwner, 0, OwnerNameLen );
  ::memcpy( m_hdrVar.pcOwner, szOwner.c_str(), OwnerNameLen );
  //m_hdrVar.tvModTime = tsModTime.getTimeVal();
  m_hdrVar.tvModTime.tv_sec = (int32_t)tsModTime.getTimeVal().tv_sec;
  m_hdrVar.tvModTime.tv_usec = (int32_t)tsModTime.getTimeVal().tv_usec;
  m_vecValues = vector<char>( getNumBytes() );
  ::memset( &m_vecValues[0], 0, getNumBytes() );
}

////////////////////////////////////////////////////////////////////////////////
// Return the value as a string.

string RtdbVariable::getValue()
{
  return getValue( 0 );
}

////////////////////////////////////////////////////////////////////////////////
// Return the indexed value as a string.

string RtdbVariable::getValue( const unsigned int &uiIndex )
{
  stringstream ssValue;
  unsigned int iOffset = getTypeSize( m_hdrVar.tType ) * uiIndex;

  switch( m_hdrVar.tType )
  {
    case Int:
      ssValue << *( (int32_t *)( &m_vecValues[iOffset] ) );
      break;
    case Real:
      ssValue << *( (double *)( &m_vecValues[iOffset] ) );
      break;
    case Char:
      ssValue << (unsigned char *)( &m_vecValues[iOffset] );
      break;
    case Bit8:
      ssValue << *( (unsigned char *)( &m_vecValues[iOffset] ) );
      break;
    case Bit16:
      ssValue << *( (uint16_t *)( &m_vecValues[iOffset] ) );
      break;
    case Bit32:
      ssValue << *( (uint32_t *)( &m_vecValues[iOffset] ) );
      break;
    case Bit64:
      ssValue << *( (uint64_t *)( &m_vecValues[iOffset] ) );
      break;
    case Pickl:
    default:
      break;
  }
  return ssValue.str();
}

////////////////////////////////////////////////////////////////////////////////
// Set the value as a string.

void RtdbVariable::setValue( const string &szValue )
{
  setValue( 0, szValue );
}

////////////////////////////////////////////////////////////////////////////////
// Set the indexed value as a string.

void RtdbVariable::setValue( const unsigned int &uiIndex,
                             const string &szValue )
{
  stringstream ssValue;
  ssValue << szValue;
  unsigned int iOffset = getTypeSize( m_hdrVar.tType ) * uiIndex;
  // Make the conversion explicit.
  int32_t iValue32;
  double xValue;
  unsigned char ucValue;
  uint16_t uiValue16;
  uint32_t uiValue32;
  uint64_t uiValue64;

  switch( m_hdrVar.tType )
  {
    case Int:
      ssValue >> iValue32;
      *( (int32_t *)( &m_vecValues[iOffset] ) ) = iValue32;
      break;
    case Real:
      ssValue >> xValue;
      *( (double *)( &m_vecValues[iOffset] ) ) = xValue;
      break;
    case Char:
      ssValue >> (unsigned char *)( &m_vecValues[0] );
      break;
    case Bit8:
      ssValue >> ucValue;
      *( (unsigned char *)( &m_vecValues[iOffset] ) ) = ucValue;
      break;
    case Bit16:
      ssValue >> uiValue16;
      *( (uint16_t *)( &m_vecValues[iOffset] ) ) = uiValue16;
      break;
    case Bit32:
      ssValue >> uiValue32;
      *( (uint32_t *)( &m_vecValues[iOffset] ) ) = uiValue32;
      break;
    case Bit64:
      ssValue >> uiValue64;
      *( (uint64_t *)( &m_vecValues[iOffset] ) ) = uiValue64;
      break;
    case Pickl:
    default:
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Return the raw header.

const RtdbVariable::Header &RtdbVariable::getHeader() const
{
  return m_hdrVar;
}

////////////////////////////////////////////////////////////////////////////////
// Return the vector of data stored here.

const vector<char> &RtdbVariable::getValues() const
{
  return m_vecValues;
}

////////////////////////////////////////////////////////////////////////////////
// The name of this RtdbVariable (may be an array).

string RtdbVariable::getName() const
{
  return string( m_hdrVar.pcName );
}

////////////////////////////////////////////////////////////////////////////////
// The total raw data size.

int RtdbVariable::getNumBytes() const
{
  int iSize = -1; // Assume error.
  iSize = getTypeSize( m_hdrVar.tType ) * m_hdrVar.iNumItems;
  if ( m_hdrVar.tType == Char )
  {
    iSize++; // +1 for '\0'.
  }
  return iSize;
}

////////////////////////////////////////////////////////////////////////////////
// What is the size (in bytes) of each approved type?

unsigned int RtdbVariable::getTypeSize( const Type &tType )
{
  int iSize = 0; // Assume error.
  switch( tType )
  {
    case Int:
      iSize = sizeof( int32_t );
      break;
    case Real:
      iSize = sizeof( double );
      break;
    case Char:
      iSize = sizeof( unsigned char );
      break;
    case Bit8:
      iSize = sizeof( unsigned char );
      break;
    case Bit16:
      iSize = sizeof( uint16_t );
      break;
    case Bit32:
      iSize = sizeof( uint32_t );
      break;
    case Bit64:
      iSize = sizeof( uint64_t );
      break;
    case Pickl:
      iSize = 0;
      break;
    default:
      iSize = 0;
      break;
  }
  return iSize;
}

////////////////////////////////////////////////////////////////////////////////
// This may be an array, so the number of items here may be > 1.

int RtdbVariable::getNumItems() const
{
  return m_hdrVar.iNumItems;
}

////////////////////////////////////////////////////////////////////////////////
// Who owns this RtdbVariable?

string RtdbVariable::getOwner() const
{
  return string( m_hdrVar.pcOwner );
}

////////////////////////////////////////////////////////////////////////////////
// One of the types enumerated above. All items are the same type.

RtdbVariable::Type RtdbVariable::getType() const
{
  return m_hdrVar.tType;
}

////////////////////////////////////////////////////////////////////////////////
// Last time this data was modified

TimeStamp RtdbVariable::getTimeStamp() const
{
  timeval tvResult;
  tvResult.tv_sec = m_hdrVar.tvModTime.tv_sec;
  tvResult.tv_usec = m_hdrVar.tvModTime.tv_usec;
  return TimeStamp( tvResult );
}

////////////////////////////////////////////////////////////////////////////////
// Return the type defined here as a string.

string RtdbVariable::getTypeString( const Type &tType )
{
  string szTypeName = "Unknown";

  switch( tType )
  {
    case Int:
      szTypeName = "Int";
      break;
    case Real:
      szTypeName = "Real";
      break;
    case Char:
      szTypeName = "Char";
      break;
    case Bit8:
      szTypeName = "Bit8";
      break;
    case Bit16:
      szTypeName = "Bit16";
      break;
    case Bit32:
      szTypeName = "Bit32";
      break;
    case Bit64:
      szTypeName = "Bit64";
      break;
    case Pickl:
      szTypeName = "Pickl";
      break;
    default:
      szTypeName = "Unknown";
      break;
  }
  return szTypeName;
}

////////////////////////////////////////////////////////////////////////////////
