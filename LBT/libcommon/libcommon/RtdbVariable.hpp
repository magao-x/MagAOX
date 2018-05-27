/// RtdbVariable.hpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef MSGD_RTDB_VARIABLE_HPP
#define MSGD_RTDB_VARIABLE_HPP

#include <vector>
#include <string>
#include <stdint.h>
#include <string.h>
#include "TimeStamp.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace msgd
{
class RtdbVariable
{
  public:
    enum Constants
    {
      VarNameLen =      80,
      OwnerNameLen =    20,
    };

  public:
    enum Type
    {
      Unknown =         0,    // Unknown type.
      Int =             1001, // Integer variable (implemented as long->int32_t)
      Real =            1002, // Real variable (implemented as double)
      Char =            1003, // Character variable. Strings are represented as arrays of characters
      Bit8 =            1004, // Byte variable (implemented as unsigned char)
      Bit16 =           1005, // Word variable (implemented as unsigned short->uint16_t)
      Bit32 =           1006, // Double word variable (implemented as unsigned long->uint32_t)
      Bit64 =           1007, // Quad word variable (implemented as unsigned long long->uint64_t)
      Pickl =           1008, // Serialized Python object (implemented as unsigned char array)
    };

  public:
    #pragma pack( push, 1)
    struct timeval32
    {
      int32_t tv_sec;
      int32_t tv_usec;
    };
    #pragma pack( pop ) // Back to whatever the previous packing mode was.

  public:
    #pragma pack( push, 1 )
    struct Header
    {
      char pcName[VarNameLen];
      char pcOwner[OwnerNameLen];
      int32_t iDummy;
      RtdbVariable::Type tType;
      int32_t iNumItems;
      struct timeval32 tvModTime;
    };
    #pragma pack( pop ) // Back to whatever the previous packing mode was.

  public:
    RtdbVariable();
    virtual ~RtdbVariable();
    RtdbVariable( const RtdbVariable &var );
    RtdbVariable( const RtdbVariable::Header &hdr,
              const std::vector<char> &vecValues );
    RtdbVariable( const std::string &szName,
                  const std::string &szOwner,
                  const RtdbVariable::Type &tType,
                  const int &iNumItems,
                  const pcf::TimeStamp &tsModTime );

  public:
    const RtdbVariable &operator=( const RtdbVariable &var );

  public:
    // Return the raw header.
    const RtdbVariable::Header &getHeader() const;
    // The name of this RtdbVariable (may be an array).
    std::string getName() const;
    // The total raw data size.
    int getNumBytes() const;
    // This may be an array, so the number of items here may be > 1.
    int getNumItems() const;
    // Who owns this RtdbVariable?
    std::string getOwner() const;
    // Last time this data was modified
    pcf::TimeStamp getTimeStamp() const;
    // On of the types enumerated above. All items are the same type.
    RtdbVariable::Type getType() const;
    // Return the type defined here as a string.
    static std::string getTypeString( const RtdbVariable::Type &tType );
    // What is the size (in bytes) of each approved type?
    static unsigned int getTypeSize( const RtdbVariable::Type &tType );
    // Return the value as a string.
    std::string getValue();
    // Return the indexed value as a string.
    std::string getValue( const unsigned int &uiIndex );
    // Return the vector of data stored here.
    const std::vector<char> &getValues() const;
    // Set the value as a string.
    void setValue( const std::string &szValue );
    // Set the indexed value as a string.
    void setValue( const unsigned int &uiIndex,
                   const std::string &szValue );

  private:
    RtdbVariable::Header m_hdrVar;
    std::vector<char> m_vecValues;

}; // class RtdbVariable
} // namespace msgd

#endif // MSGD_RTDB_VARIABLE_HPP
