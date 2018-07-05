/// Fits.hpp
///
/// @author Paul Grenz
///
/// Routines to read and write Fits files.
///
/// BITPIX data type code values for FITS images:
///
/// #define BYTE_IMG      8  /*  8-bit unsigned integers */
/// #define SHORT_IMG    16  /* 16-bit   signed integers */
/// #define LONG_IMG     32  /* 32-bit   signed integers */
/// #define LONGLONG_IMG 64  /* 64-bit   signed integers */
/// #define FLOAT_IMG   -32  /* 32-bit single precision floating point */
/// #define DOUBLE_IMG  -64  /* 64-bit double precision floating point */
///
/// The following 4 data type codes are also supported by CFITSIO:
/// #define SBYTE_IMG  10    /*  8-bit signed integers, equivalent to */
                             /*  BITPIX = 8, BSCALE = 1, BZERO = -128 */
/// #define USHORT_IMG  20   /* 16-bit unsigned integers, equivalent to */
                             /*  BITPIX = 16, BSCALE = 1, BZERO = 32768 */
/// #define ULONG_IMG   40   /* 32-bit unsigned integers, equivalent to */
                             /*  BITPIX = 32, BSCALE = 1, BZERO = 2147483648 */

////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_FITS_HPP
#define PCF_FITS_HPP

#include <stdint.h>
#include <vector>
#include <string>
#include <string.h>  // provides 'memset'.
#include "fitsio.h"
#include "zlib.h"
#include "Logger.hpp"

namespace pcf
{
class Fits
{
  // A card hold header information.
  public:
    struct Card
    {
      private:
        std::string m_szName;
        std::string m_szValue;
        std::string m_szComment;
      public:
        Card( const std::string &szName,
              const std::string &szValue ) :
          m_szName( szName ), m_szValue( szValue ) {}
        Card( const std::string &szName,
              const std::string &szValue,
              const std::string &szComment ) :
          m_szName( szName ), m_szValue( szValue ), m_szComment( szComment ) {}
        const std::string &getName() const { return m_szName; }
        const std::string &getValue() const { return m_szValue; }
        const std::string &getComment() const { return m_szComment; }
        void setName( const std::string &szName ) { m_szName = szName; }
        void setValue( const std::string &szValue ) { m_szValue = szValue; }
        void setComment( const std::string &szComment ) { m_szComment = szComment; }
    };

  // Methods
  public:
    /// Adds a new HDU to the end of the file, then sets it
    /// as the current one and returns the number of HDUs in the file.
    /// Only IMAGE HDUs are supported.
    static uint64_t addHdu( fitsfile *pHandle,
                            const int32_t &iBitPix,
                            const std::vector<uint64_t> &vecAxesDim );
    /// What is the size of our HDU?
    static uint64_t calcHduSize( const uint64_t &NumImageBytes );
    /// Close a fits file if the handle is valid, otherwise do nothing.
    static void close( fitsfile *pHandle );
    /// Check if specified FITS file exists and return its data type and dimensions.
    static void getAttributes( const std::string &szName,
                               const uint64_t &uiHdu,
                               int32_t &iBitPix,
                               std::vector<uint64_t> &vecAxesDim );
    static void getAttributes( fitsfile *pHandle,
                               const uint64_t &uiHdu,
                               int32_t &iBitPix,
                               std::vector<uint64_t> &vecAxesDim );
    /// Get the FITSIO code that describes the number of bits per pixel.
    static int32_t getBitPix( const uint8_t &tDataSample );
    static int32_t getBitPix( const char &tDataSample );
    static int32_t getBitPix( const uint16_t &tDataSample );
    static int32_t getBitPix( const int16_t &tDataSample );
    static int32_t getBitPix( const uint32_t &tDataSample );
    static int32_t getBitPix( const int32_t &tDataSample );
    static int32_t getBitPix( const int64_t &tDataSample );
    static int32_t getBitPix( const float &tDataSample );
    static int32_t getBitPix( const double &tDataSample );

    /// Get the FITSIO type that will be written to the image.
    static int32_t getType( const uint8_t &tDataSample );
    static int32_t getType( const char &tDataSample );
    static int32_t getType( const uint16_t &tDataSample );
    static int32_t getType( const int16_t &tDataSample );
    static int32_t getType( const uint32_t &tDataSample );
    static int32_t getType( const int32_t &tDataSample );
    static int32_t getType( const int64_t &tDataSample );
    static int32_t getType( const float &tDataSample );
    static int32_t getType( const double &tDataSample );

    /// Opens an existing file for read access, adding the ".fits" extension.
    /// Will throw if an error occurs, returns the handle to the fits file.
    static fitsfile *openRead( const std::string &szFilename );
    /// Opens a file for writing. The filename argument is a 'base name'
    /// in that a time stamp may be appended to it, along with the suffix '.fits'.
    static fitsfile *openWrite( const std::string &szFilename,
                                const bool &oAppendTimeStamp );
    // Completely opens and reads a fits file.
    template<typename TT> static int64_t readImage(
                          const std::string &szFilename,
                          std::vector<TT> &vecImage,
                          std::vector<Fits::Card> &vecCards,
                          const uint64_t &uiRows,
                          const uint64_t &uiCols,
                          const bool &oUseMemBuf,
                          const std::vector<unsigned char> &vecMemBuf );
    /// Sets the current HDU to the requested index.
    /// FITS HDUs are 1-based, but here the argument is zero based.
    static void setHdu( fitsfile *pHandle,
                        const uint64_t &uiHdu );
    template<typename TT> static int64_t writeImage(
                          const std::string &szFilename,
                          const std::vector<TT> &vecImage,
                          const std::vector<Fits::Card> &vecCards,
                          const std::vector<uint32_t> &vecDims,
                          const bool &oIsMemBuf,
                          std::vector<uint8_t> &vecMemBuf,
                          const bool &oUseComprBuf,
                          std::vector<uint8_t> &vecComprBuf );

}; // Class Fits
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////
/// \brief writeImage
/// Returns the size of the file written, or the size of the (compressed)
/// file saved to a memory buffer.
/// \param szFilename Fully qualified filename.
/// \param vecImage The image data to write to the file.
/// \param vecCards The name-value pairs to write to the header.
/// \param uiRows The image width.
/// \param uiCols The image height.
/// \param oUseMemBuf Should we write the data to a memory buffer?
/// \param vecMemBuf A preallocated vector to hold the data.
/// \param oUseComprBuf Should we write the data to a memory buffer?
/// \param vecComprBuf A preallocated vector to hold the compressed data.
/// It should be MemBufSize * 1.1 + 12.

template<typename TT> int64_t pcf::Fits::writeImage(
                              const std::string &szFilename,
                              const std::vector<TT> &vecImage,
                              const std::vector<Fits::Card> &vecCards,
                              const std::vector<uint32_t> &vecDims,
                              const bool &oUseMemBuf,
                              std::vector<uint8_t> &vecMemBuf,
                              const bool &oUseComprBuf,
                              std::vector<uint8_t> &vecComprBuf )
{
  // Pointer to the FITS file, defined in fitsio.h
  fitsfile *fptr;
  int32_t iStatus = 0;
  uint64_t uiSavedSize = 0;

  try
  {
    if ( oUseMemBuf == true || oUseComprBuf == true )
    {
      size_t tMemBufSize = vecMemBuf.size();
      uint8_t *pMem = (uint8_t *)( &vecMemBuf[0] );
      if ( ::fits_create_memfile( &fptr, (void**)( &( pMem ) ),
                              &tMemBufSize, 0, NULL, &iStatus) != 0 )
      {
        throw ( iStatus );
      }
    }
    else
    {
      // Delete old file if it already exists.
      ::remove( szFilename.c_str() );

      // Create the data structure for the FITS file.
      if ( ::fits_create_file( &fptr, szFilename.c_str(), &iStatus ) != 0 )
      {
        throw ( iStatus );
      }
    }

    // This code supports images with N axes.
    int32_t iNumAxes = vecDims.size();
    // Each dimension of the image.
    //uint32_t puiAxes[2] = { uiRows, uiCols };
    // The code to hold the bits per pixel.
    int32_t iBitPix = getBitPix( vecImage[0] );
    // Create the primary array image of what ever type this is.
    if ( ::fits_create_img( fptr, iBitPix, iNumAxes,
                            (long*)(&vecDims[0]), &iStatus) != 0 )
    {
      throw iStatus;
    }

    // Write all the cards to the FITS file.
    for ( uint32_t ii = 0; ii < vecCards.size(); ii++ )
    {
      // Pointers to the underlying data should not be stored.
      char *pcName = (char*)( vecCards[ii].getName().c_str() );
      char *pcValue = (char*)( vecCards[ii].getValue().c_str() );
      char *pcComment = (char*)( vecCards[ii].getComment().c_str() );
      if ( vecCards[ii].getName() == "COMMENT" )
      {
        ::fits_write_comment( fptr, pcValue, &iStatus );
      }
      else
      {
        ::fits_write_key( fptr, TSTRING, pcName, pcValue, pcComment, &iStatus );
      }
      if ( iStatus != 0 )
      {
        throw iStatus;
      }
    }

    // The first pixel is numbered 1. This is just a convention.
    int32_t iFirstPixel = 1;
    // The total number of pixels in the image. There must be at least 1.
    //int32_t iElements = uiRows * uiCols;
    int32_t iElements = 1;
    for ( uint32_t ee = 0; ee < vecDims.size(); ee++ )
      iElements *= ( vecDims[ee] > 0 ) ? ( vecDims[ee] ) : ( 1 );
    // The type for the image.
    int32_t iType = getType( vecImage[0] );
    // Write the vector of data to the image.
    if ( ::fits_write_img( fptr, iType, iFirstPixel, iElements,
                      (char *)( &vecImage[0] ), &iStatus) != 0 )
    {
      throw iStatus;
    }

    // This is the size of the file (in bytes).
    uiSavedSize = calcHduSize( iElements * sizeof( vecImage[0] ) );

    if ( oUseComprBuf == true )
    {
      // Now let's compress the data.
      Bytef *pbComprBuf = (Bytef *)( &vecComprBuf[0] );
      Bytef *pbMemBuf = (Bytef *)( &vecMemBuf[0] );
      uLongf uiComprBufLen = vecComprBuf.size();
      uLongf uiMemBufLen = uiSavedSize;
      if ( ::compress2( pbComprBuf, &uiComprBufLen,
                    pbMemBuf, uiMemBufLen, Z_BEST_SPEED ) != Z_OK )
      {
        throw DATA_COMPRESSION_ERR;
      }

      // Update the saved size.
      uiSavedSize = uiComprBufLen;
    }
  }
  catch ( const int32_t &iStatus )
  {
    // For any error messages.
    // The message will only be 30 chars max, but lets have a little extra room.
    char pcErrMsg[40];
    ::memset( pcErrMsg, 0, 40 );
    ::fits_get_errstatus( iStatus, pcErrMsg );
    pcf::Logger logErr;
    logErr << pcf::Logger::Error << "Saving FITS file: '" << szFilename << "'"
           << " ERROR: " << pcErrMsg << std::endl;
  }

  // Close the file if there was no error.
  if ( iStatus == 0 )
  {
    ::fits_close_file( fptr, &iStatus );
  }

  return ( uiSavedSize );
}

////////////////////////////////////////////////////////////////////////////////
/// \brief readImage Only supports 2 axis images.
/// \param szFilename Fully qualified filename.
/// \param vecImage The image data to read from the file.
/// \param vecCards The name-value pairs to read from the header.
/// \param uiRows The image width.
/// \param uiCols The image height.
/// \param oUseMemBuf Should we read the data from a memory buffer?
/// \param vecMemBuf A preallocated vector which has the data.

template<typename TT> int64_t pcf::Fits::readImage(
                              const std::string &szFilename,
                              std::vector<TT> &vecImage,
                              std::vector<Fits::Card> &vecCards,
                              const uint64_t &uiRows,
                              const uint64_t &uiCols,
                              const bool &oUseMemBuf,
                              const std::vector<uint8_t> &vecMemBuf )
{
  // Pointer to the FITS file, defined in fitsio.h
  fitsfile *fptr;
  int32_t iStatus = 0;

  try
  {
    if ( oUseMemBuf == true )
    {
      size_t tMemBufSize = vecMemBuf.size();
      uint8_t *pMem = (uint8_t *)( &vecMemBuf[0] );
      if ( ::fits_open_memfile( &fptr, "", READONLY, (void**)( &( pMem ) ),
                           &tMemBufSize, 0, NULL, &iStatus ) != 0 )
      {
        throw ( iStatus );
      }
    }
    else
    {
      if ( ::fits_open_file( &fptr, szFilename.c_str(), READONLY, &iStatus ) != 0 )
      {
        throw ( iStatus );
      }
    }

    // This code only supports images with two axes.
    int32_t iNumAxes = 0;
    int32_t iMaxNumAxes = 2;
    // Each dimension of the image.
    uint64_t puiAxes[2];
    // The code to hold the bits per pixel.
    int32_t iBitPix = 0;
    if ( ::fits_get_img_param( fptr, iMaxNumAxes, &iBitPix, &iNumAxes,
                               (long*)(puiAxes), &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // We need to check to see if the image is what we expect.
    if ( iNumAxes != iMaxNumAxes )
    {
      throw BAD_NAXES;
    }

    if ( iBitPix != getBitPix( vecImage[0] ) )
    {
      throw BAD_BITPIX;
    }

    if ( puiAxes[0] != uiRows || puiAxes[1] != uiCols )
    {
      throw BAD_PIX_NUM;
    }

    // The first pixel is numbered 1. This is just a convention.
    uint64_t uiFirstPixel = 1;
    // The total number of pixels in the image.
    uint64_t uiElements = uiRows * uiCols;
    // The type for the image.
    int64_t iType = getType( vecImage[0] );
    if ( ::fits_read_img( fptr, iType, uiFirstPixel, uiElements, NULL,
                          &vecImage[0], NULL, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Now populate the vector with the header data.
    int32_t iNumKeys = 0;
    if ( ::fits_get_hdrspace( fptr, &iNumKeys, NULL, &iStatus ) != 0 )
    {
      throw ( iStatus );
    }

    // Make sure the vector is at least big enough to hold them all.
    vecCards = std::vector<Fits::Card>( iNumKeys );

    char pcName[81];
    char pcValue[81];
    char pcComment[81];
    for ( int32_t ii = 1; ii <= iNumKeys; ii++ )
    {
      memset( pcName, 0, 81 );
      memset( pcValue, 0, 81 );
      memset( pcComment, 0, 81 );
      //  read the current key.
      ::fits_read_keyn( fptr, ii, pcName, pcValue, pcComment, &iStatus );
      vecCards[ii-1] = Fits::Card( pcName, pcValue, pcComment );
    }
  }
  catch ( const int32_t &iStatus )
  {
    // For any error messages.
    // The message will only be 30 chars max, but lets have a little extra room.
    char pcErrMsg[40];
    ::memset( pcErrMsg, 0, 40 );
    ::fits_get_errstatus( iStatus, pcErrMsg );
    pcf::Logger logErr;
    logErr << pcf::Logger::Error << "Reading FITS file: "
           << "'" << szFilename << "'"
           << " ERROR: " << pcErrMsg << std::endl;
  }

  // Close the file if there was no error.
  if ( iStatus == 0 )
  {
    ::fits_close_file( fptr, &iStatus );
  }

  return iStatus;
}

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_FITS_HPP

