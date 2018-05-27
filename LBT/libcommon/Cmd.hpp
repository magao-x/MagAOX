/// Cmd.hpp
///
/// @author Paul Grenz
///
/// The Cmd class represents a text-based command which is parsed and accesed
/// as a series of tokens.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_CMD_HPP
#define PCF_CMD_HPP

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class Cmd
{
  public:
    //  these are the errors defined for the class.
    enum Error
    {
      ErrNone = 0,
      ErrInvalidBool = -1,
      ErrUnknown = -9999
    };

  public:
    /// This is the type of exception that this class will throw.
    class Excep : public std::exception
    {
      public:
        Excep( const Cmd::Error &nCode ) : m_nCode( nCode ) {}
        Excep() : m_nCode( Cmd::ErrNone ) {}
        ~Excep() throw() {}
        int getCode() const
        {
          return m_nCode;
        }
        virtual const char *what() const throw()
        {
          return Cmd::getErrorMsg( m_nCode ).c_str();
        }
      private:
        int m_nCode;
    };

    // Constructor/destructor/copy constructor.
  public:
    Cmd();
    Cmd( const std::string &szText );

    // Operators.
  public:
    /// Returns a rhs version of the token at an index.
    const std::string &operator[]( const unsigned int &nIndex ) const;
    /// Returns a lhs version of the token at an index.
    std::string &operator[]( const unsigned int &nIndex );

    // Methods
  public:
    /// Returns the token at an index.
    template <typename TT> TT at( const unsigned int &nIndex ) const;
    /// Return an message associated with this error.
    static std::string getErrorMsg( const int &nError );
    /// Get the original line of text (no control chars).
    std::string getString() const;
    ///  merges multiple spaces, '\r', '\n' and/or '\t' into one space.
    static std::string mergeWhitespace( const std::string &szUnmerged );
    /// Returns the number of tokens parsed.
    unsigned int size() const;
    /// Split up the line into tokens based on delimitor (spaces are the default).
    /// Text in quotes ("") is treated as one token.
    /// Returns the position where an error occurred, or string::npos if none.
    static unsigned int parse( const std::string &szLine,
                               std::vector<std::string> &vecTokens,
                               const char &cDelim = ' ' );
    /// Replaces ACK, BEL, CR, LF with printable versions:
    /// "<ACK>", "<BEL>", "<CR>", "<LF>".
    static std::string replaceNonPrintable( const std::string &szText );
    /// Breaks a string up into tokens based on a delimitor.
    static std::vector<std::string> tokenize( const std::string &szList,
        const std::string &szDelim = std::string( " " ) );
    /// Returns a string version of a object.
    template <typename TT> static std::string toString( const TT &tType );
    /// Removes whitespace chars from the front and rear of a string.
    static std::string trimWhitespace( const std::string &szPadded );
    /// Replaces all occurances of a string in another string.
    static std::string replaceAll( const std::string &szSource,
                                   const std::string &szFind,
                                   const std::string &szReplace );

  private:
    std::vector<std::string> m_vecTokens;

}; // Class Cmd
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////
/// Returns a string version of a object.

namespace pcf
{
template <typename TT> std::string Cmd::toString( const TT &tType )
{
  std::stringstream ssItem;
  ssItem.precision( 20 );
  ssItem << std::boolalpha << tType;
  return ssItem.str();
}
}

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
template <typename TT> TT Cmd::at( const unsigned int &uiIndex ) const
{
  std::stringstream ssItem;
  TT tItem;
  ssItem.str( m_vecTokens[ uiIndex ] );
  ssItem.precision( 20 );
  ssItem >> std::boolalpha >> tItem;
  return tItem;
}
}

////////////////////////////////////////////////////////////////////////////////
/// Booleans can be a bit tricky, because people like to put "yes", "no",
/// "on", "off", "true", "false", etc.

namespace pcf
{
template <> inline bool Cmd::at<bool>( const unsigned int &uiIndex ) const
{
  std::string szToken = m_vecTokens[ uiIndex ];
  std::transform( szToken.begin(), szToken.end(), szToken.begin(), ::tolower );

  // We may have something other than "true" or "false".
  if ( szToken == "true" || szToken == "on" || szToken == "yes" )
    return true;
  else if ( szToken == "false" || szToken == "off" || szToken == "no" )
    return false;
  else
    throw Cmd::Excep( Cmd::ErrInvalidBool );
}
}

////////////////////////////////////////////////////////////////////////////////
/// Strings can be a bit tricky, because we want the exact string....
/// spaces and all.

namespace pcf
{
template <> inline std::string Cmd::at<std::string>( const unsigned int &uiIndex ) const
{
  return m_vecTokens[ uiIndex ];
}
}

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_CMD_HPP
