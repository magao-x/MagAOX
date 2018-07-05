/// Fits.hpp
///
/// @author Paul Grenz
////////////////////////////////////////////////////////////////////////////////

#include <exception>
#include <algorithm>    // std::replace
#include "TimeStamp.hpp"
#include "Fits.hpp"

using std::runtime_error;
using std::string;
using std::vector;
using std::endl;
using pcf::TimeStamp;
using pcf::Logger;
using pcf::Fits;

////////////////////////////////////////////////////////////////////////////////
// What is the size of our HDU?

uint64_t Fits::calcHduSize( const uint64_t &uiNumImageBytes )
{
  // The data will be the smallest amount that is a multiple of 2880.
  uint64_t uiHduSize = uiNumImageBytes;
  if ( uiNumImageBytes % 2880 != 0 )
  {
    uiHduSize = ( ( uiNumImageBytes / 2880 ) + 1 ) * 2880;
  }
  // Assume that there is only one header (size = 2880).
  uiHduSize += 2880;

  return uiHduSize;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Fits::openRead Opens an existing file for read access.
/// \param szFilename The file name to open, not assumed to have the '.fits'
/// extension.
/// \return The handle to the fits file.

fitsfile *Fits::openRead( const string &szFilename )
{
  try
  {
    int32_t iStatus = 0;
    fitsfile *pHandle;
    string szName = szFilename + ".fits";

    // Open file for reading
    if ( ::fits_open_file( &pHandle, szName.c_str(), READONLY, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    return pHandle;
  }
  catch ( const int32_t &iStatus )
  {
    char pcErrMsg[40];
    ::memset( pcErrMsg, 0, 40 );
    ::fits_get_errstatus( iStatus, pcErrMsg ) ;
    throw ( runtime_error( string( "Fits::openRead ERROR: " ) + pcErrMsg ) );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Fits::openWrite Opens a file for writing. The filename
/// argument is a 'base name' in that a time stamp may be appended to it,
/// along with the suffix '.fits'.
/// \param szFilename The base file name to use.
/// \param oAppendTimeStamp Should a time stamp be added to the end of the name?
/// \return the handle to the fits file.

fitsfile *Fits::openWrite( const string &szFilename,
                           const bool &oAppendTimeStamp )
{
  try
  {
    int32_t iStatus = 0;
    fitsfile *pHandle;
    // We possibly need the date appended to the file name as a number.
    string szName = szFilename;
    if ( oAppendTimeStamp == true )
    {
      string szIso8601Mod = TimeStamp::now().getFormattedIso8601Str();
      //std::replace( szIso8601Mod.begin(), szIso8601Mod.end(), '-', '_' );
      std::replace( szIso8601Mod.begin(), szIso8601Mod.end(), 'T', '_' );
      //std::replace( szIso8601Mod.begin(), szIso8601Mod.end(), ':', '_' );
      //std::replace( szIso8601Mod.begin(), szIso8601Mod.end(), '.', '_' );
      szName += "_" + szIso8601Mod;
    }
    szName += ".fits";

    Logger logMsg;
    logMsg << Logger::Debug << "Filename to create: " << szName << endl;

    // Create file for writing.
    if ( ::fits_create_file( &pHandle, szName.c_str(), &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    return pHandle;
  }
  catch ( const int32_t &iStatus )
  {
    char pcErrMsg[40];
    ::memset( pcErrMsg, 0, 40 );
    ::fits_get_errstatus( iStatus, pcErrMsg ) ;
    throw ( runtime_error( string( "Fits::openWrite ERROR: " ) + pcErrMsg ) );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Fits::close Close a fits file if the handle is valid, otherwise
/// do nothing.
/// \param pHandle The handle to the fits file.

void Fits::close( fitsfile *pHandle )
{
  try
  {
    int32_t iStatus = 0;

    if ( pHandle == NULL )
    {
      return;
    }

    if ( ::fits_close_file( pHandle, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }
  }
  catch ( const int32_t &iStatus )
  {
    char pcErrMsg[40];
    ::memset( pcErrMsg, 0, 40 );
    ::fits_get_errstatus( iStatus, pcErrMsg ) ;
    throw ( runtime_error( string( "Fits::close ERROR: " ) + pcErrMsg ) );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Fits::addHdu Adds a new HDU to the end of the file, then sets it
/// as the current one. Only IMAGE HDUs are supported.
/// \param [in] pHandle Fitsio file handle.
/// \param [in] iBitPix Fitsio defined data type.
/// \param [in] vecAxesDim The vector of axis dimensions.
/// \return The number of HDUs in the file.

uint64_t Fits::addHdu( fitsfile *pHandle,
                       const int32_t &iBitPix,
                       const vector<uint64_t> &vecAxesDim )
{
  try
  {
    int32_t iStatus = 0;
    // The number of dimensions.
    int32_t iNumAxes = (int32_t)( vecAxesDim.size() );
    // We need the array to be of type long.
    long *puiAxesDim = (long*)( &vecAxesDim[0] );
    // This will hold the number of HDUs in the FITS file.
    int32_t iHdus = 0;

    if ( pHandle == NULL )
    {
      throw ( BAD_FILEPTR );
    }

    // How many HDUs do we have now?
    if ( ::fits_get_num_hdus( pHandle, &iHdus, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Increment the HDU.
    iHdus++;

    // Add one to the end.
    if ( ::fits_create_img( pHandle, iBitPix, iNumAxes,
                            puiAxesDim, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Try to set the current HDU to the last one we just created.
    // In cfitsio, HDUs are 1-based, so the count and last index are the same.
    if ( ::fits_movabs_hdu( pHandle, iHdus, NULL, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    return (uint64_t)( iHdus );
  }
  catch ( const int32_t &iStatus )
  {
    char pcErrMsg[40];
    ::memset( pcErrMsg, 0, 40 );
    ::fits_get_errstatus( iStatus, pcErrMsg ) ;
    throw ( runtime_error( string( "Fits::addHdu ERROR: " ) + pcErrMsg ) );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Fits::setHdu Sets the current HDU to the requested index.
/// FITS HDUs are 1-based, but here
/// the argument is zero based.
/// \param [in] pHandle Fitsio file handle.
/// \param [in] uiHdu The 0-based requested index.

void Fits::setHdu( fitsfile *pHandle,
                   const uint64_t &uiHdu )
{
  try
  {
    int32_t iStatus = 0;
    // The number of HDUs contained in this FITS file.
    int32_t iHdus = 0;

    if ( pHandle == NULL )
    {
      throw ( BAD_FILEPTR );
    }

    // How many HDU's do we have?
    if ( ::fits_get_num_hdus( pHandle, &iHdus, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Our requested index is zero-based, so it must be less than the count.
    if ( uiHdu >= (uint64_t)( iHdus ) )
    {
      throw ( BAD_HDU_NUM );
    }

    // Try to set the current HDU.
    if ( ::fits_movabs_hdu( pHandle, uiHdu+1, NULL, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }
  }
  catch ( const int64_t &iStatus )
  {
    char pcErrMsg[40];
    ::memset( pcErrMsg, 0, 40 );
    ::fits_get_errstatus( iStatus, pcErrMsg ) ;
    throw ( runtime_error( string( "Fits::setHdu ERROR: " ) + pcErrMsg ) );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief FitsStream::getAttributes Check if specified FITS file exists
/// and return the requested HDU attributes. Assumes that the HDU is of
/// type IMAGE. Note: The file name is a full path name with suffix.
/// The number of columns for a Vector type is always 1. Data is stored in
/// FITS files in column major order (row index changes most rapidly).
/// \param [in] szName file name.
/// \param [in] uiHdu The hdu to request the info from. Zero-based.
/// \param [out] tType data type.
/// \param [out] uiRows number of rows.
/// \param [out] uiCols number of columns.
/// \param [out] uiFrames number of frames.
/// \return True if file exists and has expected format.

void Fits::getAttributes( const string &szName,
                          const uint64_t &uiHdu,
                          int32_t &iBitPix,
                          vector<uint64_t> &vecAxesDim )
{
  try
  {
    int32_t iStatus = 0;
    int32_t iNumAxes = 0;
    vecAxesDim.resize( 100, 0 );
    fitsfile *pHandle;

    // Open file
    if ( ::fits_open_file( &pHandle, szName.c_str(), READONLY, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Try to set the current HDU to the desired one.
    if ( ::fits_movabs_hdu( pHandle, uiHdu+1, NULL, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Get BITPIX, NAXIS1, NAXIS2, and NAXIS3 keywords
    if ( ::fits_get_img_param( pHandle, 100, &iBitPix, &iNumAxes,
                               (long*)(&( vecAxesDim[0] )), &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Resize the vector to the exact size.
    vecAxesDim.resize( iNumAxes );

    ::fits_close_file( pHandle, &iStatus );
  }
  catch ( const int32_t &iStatus )
  {
    char pcErrMsg[40];
    ::memset( pcErrMsg, 0, 40 );
    ::fits_get_errstatus( iStatus, pcErrMsg ) ;
    throw ( runtime_error( string( "Fits::getAttributes ERROR: " ) + pcErrMsg ) );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief FitsStream::getAttributes Check if specified FITS file exists
/// and return the requested HDU attributes. Assumes that the HDU is of
/// type IMAGE. Note: The file name is a full path name with suffix.
/// The number of columns for a Vector type is always 1. Data is stored in
/// FITS files in column major order (row index changes most rapidly).
/// \param [in] pHandle The handle to the previously opened FITS file.
/// \param [in] uiHdu The hdu to request the info from. Zero-based.
/// \param [out] tType data type.
/// \param [out] uiRows number of rows.
/// \param [out] uiCols number of columns.
/// \param [out] uiFrames number of frames.

void Fits::getAttributes( fitsfile *pHandle,
                          const uint64_t &uiHdu,
                          int32_t &iBitPix,
                          vector<uint64_t> &vecAxesDim )
{
  try
  {
    int32_t iStatus = 0;
    int32_t iNumAxes = 0;
    vecAxesDim.resize( 100, 0 );

    if ( pHandle == NULL )
    {
      throw ( BAD_FILEPTR );
    }

    // Try to set the current HDU to the desired one.
    if ( ::fits_movabs_hdu( pHandle, uiHdu+1, NULL, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Get BITPIX, NAXIS1, NAXIS2, and NAXIS3 keywords
    if ( ::fits_get_img_param( pHandle, 100, &iBitPix, &iNumAxes,
                               (long*)(&( vecAxesDim[0] )), &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Resize the vector to the exact size.
    vecAxesDim.resize( iNumAxes );
  }
  catch ( const int32_t &iStatus )
  {
    char pcErrMsg[40];
    ::memset( pcErrMsg, 0, 40 );
    ::fits_get_errstatus( iStatus, pcErrMsg ) ;
    throw ( runtime_error( string( "Fits::getAttributes ERROR: " ) + pcErrMsg ) );
  }
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getBitPix( const uint8_t &tDataSample )
{
  return BYTE_IMG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getBitPix( const char &tDataSample )
{
  return SBYTE_IMG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getBitPix( const uint16_t &tDataSample )
{
  return USHORT_IMG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getBitPix( const int16_t &tDataSample )
{
  return SHORT_IMG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getBitPix( const uint32_t &tDataSample )
{
  return ULONG_IMG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getBitPix( const int32_t &tDataSample )
{
  return LONG_IMG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getBitPix( const int64_t &tDataSample )
{
  return LONGLONG_IMG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getBitPix( const float &tDataSample )
{
  return FLOAT_IMG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getBitPix( const double &tDataSample )
{
  return DOUBLE_IMG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getType( const uint8_t &tDataSample )
{
  return TBYTE;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getType( const char &tDataSample )
{
  return TSBYTE;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getType( const uint16_t &tDataSample )
{
  return TUSHORT;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getType( const int16_t &tDataSample )
{
  return TSHORT;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getType( const uint32_t &tDataSample )
{
  return TUINT;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getType( const int32_t &tDataSample )
{
  return TINT;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getType( const int64_t &tDataSample )
{
  return TLONGLONG;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getType( const float &tDataSample )
{
  return TFLOAT;
}

////////////////////////////////////////////////////////////////////////////////

int32_t Fits::getType( const double &tDataSample )
{
  return TDOUBLE;
}

////////////////////////////////////////////////////////////////////////////////
