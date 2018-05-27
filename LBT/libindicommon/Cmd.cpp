/// Cmd.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <sstream>
#include <algorithm>
#include <iostream>
#include "Cmd.hpp"

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::stringstream;
using pcf::Cmd;

////////////////////////////////////////////////////////////////////////////////

Cmd::Cmd()
{
}

////////////////////////////////////////////////////////////////////////////////

unsigned int Cmd::size() const
{
  return m_vecTokens.size();
}

////////////////////////////////////////////////////////////////////////////////

Cmd::Cmd( const string &szText )
{
  m_vecTokens.clear();

  string szTextMod = trimWhitespace( szText );
  szTextMod = mergeWhitespace( szTextMod );

  parse( szTextMod, m_vecTokens );
}

////////////////////////////////////////////////////////////////////////////////

const string &Cmd::operator[]( const unsigned int &uiIndex ) const
{
  return m_vecTokens[ uiIndex ];
}

////////////////////////////////////////////////////////////////////////////////
/*
const string& Cmd::at( const unsigned int& uiIndex ) const
{
  return m_vecTokens[ uiIndex ];
}
*/
////////////////////////////////////////////////////////////////////////////////

string &Cmd::operator[]( const unsigned int &uiIndex )
{
  return m_vecTokens[ uiIndex ];
}

////////////////////////////////////////////////////////////////////////////////
/*
string& Cmd::at( const unsigned int& uiIndex )
{
  return m_vecTokens[ uiIndex ];
}
*/
////////////////////////////////////////////////////////////////////////////////

string Cmd::getErrorMsg( const int &nError )
{
  switch ( nError )
  {
    case ErrInvalidBool:
      return "value was not 'on', 'off', 'yes', 'no', 'true' or 'false'";
      break;
    default:
      return "Unknown error";
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////

string Cmd::getString() const
{
  string szText;
  for ( unsigned int ii = 0; ii < m_vecTokens.size(); ii++ )
  {
    // Do we need to output quotes around the token?
    size_t nIndex = m_vecTokens[ii].find( ' ' );
    if ( nIndex != string::npos )
      szText += '"';
    szText += m_vecTokens[ii];
    if ( nIndex != string::npos )
      szText += '"';
    szText += " ";
  }
  return szText;
}

////////////////////////////////////////////////////////////////////////////////
///  merges multiple spaces, '\r', '\n' and/or '\t' into one space.

string Cmd::mergeWhitespace( const string &szUnmerged )
{
  // Make a copy of the unmerged string to modify.
  string szNew( szUnmerged );

  // Replace all instances of '\t', '\n', '\r' with ' '.
  std::replace( szNew.begin(), szNew.end(), '\t', ' ' );
  std::replace( szNew.begin(), szNew.end(), '\r', ' ' );
  std::replace( szNew.begin(), szNew.end(), '\n', ' ' );

  string szFind( "  " );
  string szReplace( " " );
  int nFindSize = szFind.size();

  unsigned int unPos = ( unsigned int )( string::npos );
  while ( ( unPos = szNew.find( szFind ) ) != ( unsigned int )( string::npos ) )
  {
    szNew.replace( unPos, nFindSize, szReplace );
  }

  return szNew;
}

////////////////////////////////////////////////////////////////////////////////
///  breaks a string up into tokens based on a delimitor.

vector<string> Cmd::tokenize( const string &szList,
                              const string &szDelim )
{
  vector<string> vecTokens;

  // Do we have a delimited list?
  if ( szList.size() > 0 )
  {
    unsigned int nStart = 0;
    unsigned int nStop = 0;
    int nDelimSize = szDelim.size();
    string szItem;

    // Repeat until we cannot find another delimitor.
    while ( ( nStop = szList.find( szDelim, nStart ) ) !=
            ( unsigned int )( string::npos ) )
    {
      // Pull out the token.
      szItem = szList.substr( nStart, nStop - nStart );
      vecTokens.push_back( szItem );
      // Set the new starting position after the delimitor.
      nStart = nStop + nDelimSize;
    }
    // Are there any chars left after the last delim?
    if ( nStop == ( unsigned int )( string::npos ) && nStart < szList.size() )
    {
      // There are chars after the last delim - this is the last token.
      szItem = szList.substr( nStart, szList.size() - nStart );
      vecTokens.push_back( szItem );
    }
  }
  return vecTokens;
}

////////////////////////////////////////////////////////////////////////////////
///  removes whitespace chars from the front and rear of a string.

string Cmd::trimWhitespace( const string &szPadded )
{
  string szNew( szPadded );

  // Search from the front.
  szNew.erase( 0, szNew.find_first_not_of( " \t\r\n" ) );

  // Search from the back.
  szNew.erase( szNew.find_last_not_of( " \t\r\n" ) + 1 );

  return szNew;
}

////////////////////////////////////////////////////////////////////////////////
/// Split up the line into tokens based on a delimitor (a space is the default).
/// Text in quotes ("") is treated as one token.
/// Returns the position where an error occurred, or string::npos if none.

unsigned int Cmd::parse( const string &szLine,
                         vector<string> &vecTokens,
                         const char &cDelim )
{
  // Make sure we are starting fresh.
  vecTokens.clear();

  unsigned int uiErrPos = ( unsigned int )( string::npos );
  char cCurr = 0;         // The current char.
  string szCurrToken;     // The current token.
  bool oInQuote = false;
  bool oHitComment = false;
  bool oEscapeNext = false;

  unsigned int ii = 0;
  for ( ; ii < szLine.size() && oHitComment == false; ii++ )
  {
    // Get the current char.
    cCurr = szLine.at( ii );

    // Are we going to escape the next char?
    if ( cCurr == '\\' && oInQuote == true )
    {
      // Is this another slash (escaped slash)?
      if ( oEscapeNext != true )
        oEscapeNext = true;
      else
      {
        // This is simply an escaped slash.
        szCurrToken += cCurr;
        oEscapeNext = false;
      }
    }

    // Are we possibly leaving a section of quoted text?
    else if ( cCurr == '"' && oInQuote == true )
    {
      // Are we escaping the quote or ending the string?
      if ( oEscapeNext != true )
        oInQuote = false;
      else
      {
        szCurrToken += cCurr;
        oEscapeNext = false;
      }
    }

    // Are we possibly entering a section of quoted text?
    else if ( cCurr == '"' && oInQuote == false )
    {
      oInQuote = true;
    }

    // If we are in a quote, just add the char.
    else if ( oInQuote == true )
    {
      szCurrToken += cCurr;
      oEscapeNext = false;
    }

    // Is ch a #? this is a line terminator.
    else if ( cCurr == '#' )
    {
      oHitComment = true;
    }

    // Is ch the token delimitor? This is a space by default.
    else if ( cCurr == cDelim )
    {
      // Add the current token. It should be added even if it has zero
      // length. This way the other tokens are in the correct positions.
      vecTokens.push_back( szCurrToken );
      szCurrToken.erase();
    }

    // Otherwise, just add cCurr to the curr token.
    else
    {
      szCurrToken += cCurr;
      oEscapeNext = false;
    }
  }

  // Make sure the last token was added.
  if ( szCurrToken.size() > 0 )
  {
    vecTokens.push_back( szCurrToken );
    szCurrToken.erase();
  }

  // Make sure we do not have a dangling quote.
  if ( oInQuote == true )
    uiErrPos = ii;

  return uiErrPos;
}

////////////////////////////////////////////////////////////////////////////////
/// Modifies the string 'szText' such that any unprintable chars are replaced
/// with a human-readable sequence. The original text is not modified.
/// ACK -> '<ACK>'
/// BEL -> '<BEL>'
/// CR  -> '<CR>'
/// LF  -> '<LF>'
/// @param szText The text to be modified.
/// @return The modified text.

string Cmd::replaceNonPrintable( const string &szText )
{
  // We keep a count of how many are replaced. This may be useful later.
  //uiNumAck = 0;
  //uiNumBel = 0;
  //uiNumCr = 0;
  //uiNumLf = 0;

  int nAck = 6;
  int nBel = 7;
  int nCr = 13;
  int nLf = 10;

  string szModified( szText );

  // ACK
  size_t pp = 0;
  size_t rr = 0;
  while ( ( rr = szModified.find( nAck, pp ) ) != string::npos )
  {
    szModified.replace( rr, 1, "<ACK>" );
    //uiNumAck++;
    pp = rr + 1;
  }

  // BEL
  pp = 0;
  rr = 0;
  while ( ( rr = szModified.find( nBel, pp ) ) != string::npos )
  {
    szModified.replace( rr, 1, "<BEL>" );
    //uiNumBel++;
    pp = rr + 1;
  }

  // CR
  pp = 0;
  rr = 0;
  while ( ( rr = szModified.find( nCr, pp ) ) != string::npos )
  {
    szModified.replace( rr, 1, "<CR>" );
    //uiNumCr++;
    pp = rr + 1;
  }

  // LF
  pp = 0;
  rr = 0;
  while ( ( rr = szModified.find( nLf, pp ) ) != string::npos )
  {
    szModified.replace( rr, 1, "<LF>" );
    //uiNumLf++;
    pp = rr + 1;
  }

  return szModified;
}

////////////////////////////////////////////////////////////////////////////////
/// Replaces all occurances of a string in another string.

string Cmd::replaceAll( const string &szSource,
                        const string &szFind,
                        const string &szReplace )
{
  string szReplaced = szSource;

  // Find the first instance.
  size_t tPos = szReplaced.find( szFind );

  // Repeat the find until we reach the end.
  while ( tPos != std::string::npos )
  {
    // Replace this instance of szFind.
    szReplaced.replace( tPos, szFind.size(), szReplace );
    // Find the next instance from the end of the replaced text.
    tPos = szSource.find( szFind, tPos + szReplace.size() );
  }

  return szReplaced;
}

////////////////////////////////////////////////////////////////////////////////
