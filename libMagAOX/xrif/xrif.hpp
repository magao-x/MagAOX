/** \file xrif.hpp
  * \brief The MagAO-X Raw-Image Format
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup xrif_files
  */

#ifndef xrif_xrif_hpp
#define xrif_xrif_hpp

#include <ImageStreamIO.h>


namespace MagAOX
{
namespace xrif
{

/// The type used for storing the width and height dimensions of images.
typedef uint32_t dimensionT;

/// The type used for storing the ImageStreamIO data type code.
typedef uint8_t typeCodeT;
   
/// The type used for storing the time stamp seconds field
typedef uint64_t timeSecT;

/// The type used for storing the time stamp nano-seconds field.
typedef uint32_t timeNSecT;

/// Get the size of a single raw frame including the timestamp.
/**
  * \returns the size of a raw image frame 
  */  
inline
size_t rawImageFrameSz( size_t width,  ///< [in] width of a single image
                        size_t height, ///< [in] height of a single image
                        size_t dataSz  ///< [in] the size of the image data type
                      )
{
   return sizeof(timeSecT) + sizeof(timeNSecT) + width*height*dataSz;
}

/// Get the offset to the start of a buffer of raw frames, i.e. the length of the header which includes the width, height, and data type at the beginning.
/**
  * \returns the size of a raw imaget buffer
  */  
inline
size_t rawImageBufferStart()
{
   return sizeof(dimensionT) + sizeof(dimensionT) + sizeof(typeCodeT);
}

/// Get a pointer to an image frame within a buffer, relative the end of the buffer header.
/**
  * \returns a pointer to the first byte of a frame within a buffer, which is the time stamp.
  */ 
inline
char * rawImageFrameStart( size_t imNo,
                           char * bufferStart,
                           size_t width,
                           size_t height,
                           size_t dataSz
                         )
{
   return bufferStart + imNo*rawImageFrameSz(width, height, dataSz);
}

/// Get a pointer to an image within a buffer, relative the end of the buffer header.
/**
  * \returns a pointer to the first byte of an image within a buffer, past the time stamp
  */ 
inline
char * rawImageStart( size_t imNo,
                      char * bufferStart,
                      size_t width,
                      size_t height,
                      size_t dataSz
                    )
{
   return rawImageFrameStart(imNo, bufferStart, width, height, dataSz) + sizeof(timeSecT) + sizeof(timeNSecT);
}

/// Get the timespec from a raw image frame
int getTimeSpec( timespec & ts,
                 size_t imNo,
                 char * bufferStart,
                 size_t width,
                 size_t height,
                 size_t dataSz
               )
{
   //Casts to char for pointer arithmetic
   char * frameStart = rawImageFrameStart(imNo, bufferStart, width, height, dataSz);
   
   ts.tv_sec = *((timeSecT *) frameStart);
   ts.tv_nsec = *((timeNSecT *) (frameStart + sizeof(timeSecT)));
   
   return 0;
}

int copyRawImageFrame( size_t imNo,
                       char * bufferStart,
                       const char * imageFrame,
                       timeSecT timeSec,
                       timeNSecT timeNSec,
                       size_t width,
                       size_t height,
                       size_t dataSz
                     )
{
   //Casts to char for pointer arithmetic in bytes
   char * frameStart = rawImageFrameStart(imNo, bufferStart, width, height, dataSz);
   
   *((timeSecT *) frameStart) = timeSec;
   *((timeNSecT *) (frameStart + sizeof(timeSecT))) = timeNSec;
   
   memcpy( frameStart + sizeof(timeSecT) + sizeof(timeNSecT), imageFrame, width*height*dataSz);
   
   return 0;
}
   
/// Get the filename timestamp for a timespec.
/** Fills in a string with the timestamp encoded as
  * \verbatim
    YYYYMMDDHHMMSSNNNNNNNNN
    \endverbatim
  *
  */ 
int getTimeStamp( char * tstamp, ///< [out] the string to hold the formatted time.  Must be at least 24 bytes in length.
                  const timespec & ts
                )
{
   tm uttime;//The broken down time.
   
   if(gmtime_r(&ts.tv_sec, &uttime) == 0)
   {
      std::cerr << "Error getting UT time (gmtime_r returned 0). At: " <<  __FILE__ << " " << __LINE__ << "\n";
      return -1;
   }
   
   snprintf(tstamp, 24, "%04i%02i%02i%02i%02i%02i%09i", uttime.tm_year+1900, uttime.tm_mon+1, uttime.tm_mday, uttime.tm_hour, uttime.tm_min, uttime.tm_sec, static_cast<int>(ts.tv_nsec)); 
   
   return 0;   
}

int addTimeStamp( char * fname,
                  size_t tsOffset,
                  size_t chunkStart,
                  char * rawImageBuffStart,
                  size_t width,
                  size_t height,
                  size_t byteDepth 
                )
{
   timespec ts;
   
   getTimeSpec( ts, chunkStart, rawImageBuffStart, width, height, byteDepth);

   char * tstamp = fname + tsOffset;

   getTimeStamp(tstamp, ts);
   tstamp[23] = '.';
   
   return 0;
}

ssize_t writeBuffer( char * fname,
                     size_t tsOffset,
                     size_t chunkStart,
                     size_t chunkLength,
                     char * rawImageBuffStart,
                     size_t width,
                     size_t height,
                     uint8_t typeCode
                   )
{
   size_t byteDepth = ImageStreamIO_typesize(typeCode);
   
   addTimeStamp( fname, tsOffset, chunkStart, rawImageBuffStart, width, height, byteDepth);
                        
   int fd = open(fname, O_WRONLY | O_CREAT, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
   
   if(fd < 0) perror("open"); ///\todo log this better
   
   char * frameStart = rawImageFrameStart( chunkStart, rawImageBuffStart, width, height, byteDepth); 
   
   size_t bytesToWrite = chunkLength*rawImageFrameSz(width, height, byteDepth);

   dimensionT fWidth = width;
   dimensionT fHeight = height;
   typeCodeT fCode = typeCode;
   
   ssize_t totw = 0;

   ///\toto pack header in single buffer for 1 write   
   ///\todo write error checks
   ssize_t bw = write(fd, &fWidth, sizeof(dimensionT));
   
   totw += bw;
   
   bw = write(fd, &fHeight, sizeof(dimensionT));
   
   totw += bw;
   
   bw = write(fd, &fCode, sizeof(typeCodeT));
   
   totw += bw;
   
   bw = write(fd, frameStart, bytesToWrite);
   
   totw += bw;
   
   if(close(fd) < 0) perror("close"); ///\todo log this more better
   
   return totw;
}

                
} //namespace xrif
} //namespace MagAO-X

#endif //xrif_xrif_hpp
