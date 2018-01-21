/** \file logFile.hpp 
  * \brief Manage an HDF5 log file.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-08-28 created by JRM
  */ 

#ifndef logger_logFile_hpp
#define logger_logFile_hpp


#include <iostream>

#include <string>


#include "/home/jrmales/Source/c/mxlib/include/stringUtils.hpp"


#include "../utils/H5Utils.hpp"
#include "../time/timespecX.hpp"

namespace MagAOX
{
   
using namespace MagAOX::utils;

namespace logger
{

/// A class to manage HDF5 log files
/** Manages an HDF5 file containing MagAO-X logs.
  *
  * The log entries are written as a binary stream in HDF5 datasets of a configurable
  * maximum size.  If this size will be exceed by the next entry, then a new set is created.
  *
  * Similarly, files have a maximum size at which point the current file is closed and a new one is created.
  *
  * Filenames have a standard form of: [path]/[name]_YYYYMMDDHHMMSSNNNNNNNNN.[ext] where fields in [] are configurable.
  * 
  * The timestamp is from the first entry of the file.  Similarly, each dataset has attributes recording the timestamp (in sec 
  * and nsec) of the first entry in that dataset.
  * 
  */
class logFile
{
   
protected:
   
   /** \name Configurable Parameters
     *@{
     */ 
   std::string _path{"."}; ///< The base path for the log files.
   std::string _name{"xlog"}; ///< The base name for the log files.
   std::string _ext{"xlog"}; ///< The extension for the log files.

   hsize_t _chunkSize{512}; ///< The chunk size. Default is 256 B.  \todo optimize chunk size in HDF5 log files
   
   hsize_t _maxSetSize{1048576}; ///< The maximum dataset size.  Default is 1 MB.
   hsize_t _maxFileSize{10485760}; ///< The maximum file size (of the data, not including HDF5 overhead and attributes). Default is 10 MB.
   ///@}
   
   /** \name Internal State
     *@{
     */ 
   hsize_t _currFileSize{0}; ///< The current file size.
   hsize_t _currSet{0};    ///< The current dataset number.
   hsize_t _currSetSize{0}; ///< The current dataset size.
   
   H5Handle_F _file_H; ///< The open file handle.
   H5Handle_D _dataSet_H; ///< The open dataset handle.
  

   ///@}
   
public:
   
   /// Default constructor
   /** Currently does nothing.
     */
   logFile();

   ///Destructor
   /** Closes all open hdf5 pointers
     */
   ~logFile();
   
   /// Set the path.
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int path( const std::string & newPath /**< [in] the new value of _path */ );
   
   /// Get the path.
   /**
     * \returns the current value of _path.
     */
   std::string path();
   
   /// Set the log name
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int name( const std::string & newName /**< [in] the new value of _name */ );
   
   /// Get the name
   /**
     * \returns the current value of _name. 
     */
   std::string name();
   
   /// Set the chunk size
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int chunkSize( hsize_t newChunkSize /**< [in] the new value of _chunkSize */ );
   
   /// Get the chunk size
   /**
     * \returns the current value of _chunkSize
     */
   hsize_t chunkSize();
   
   /// Set the maximum dataset size
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int maxSetSize( hsize_t newMaxSetSize /**< [in] the new value of _maxSetSize */);
   
   /// Get the maximum dataset size
   /**
     * \return the current value of _maxSetSize
     */
   hsize_t maxSetSize();
   
   /// Set the maximum file size
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int maxFileSize( hsize_t newMaxFileSize/**< [in] the new value of _maxFileSize */);
   
   /// Get the maximum file size
   /**
     * \returns the current value of _maxFileSize
     */
   hsize_t maxFileSize();

   ///Write a log entry to the file
   /**
     * \todo document this
     * 
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int writeLog( std::shared_ptr<char> & data, hsize_t N, time::timespecX & ts);

   ///Close all H5 library pointers
   /**
     * \todo investigate library shutdown
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   int closeAll();

protected:
   
   ///Create a new file
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int createFile(time::timespecX & ts /**< [in] A MagAOX timespec, used to set the timestamp */);
   
   ///Create a new data space
   /**
     * \todo we maybe don't need a class-wide space handle, perhaps have this return the handle.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   hid_t createSpace(hsize_t N /**< [in] the size of the space */);
   
   ///Create a new data set
   /** Writes an attribute containing the the timestamp (sec and nsec) of the first entry to be written to this set.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */   
   int createSet(time::timespecX & ts /**< [in] the timestamp of the first entry in the set */);

};


inline
logFile::logFile()
{
}

inline
logFile::~logFile()
{
   closeAll();
}

inline
int logFile::path( const std::string & newPath)
{
   _path = newPath;
   return 0;
}

inline
std::string logFile::path()
{
   return _path;
}

inline
int logFile::name( const std::string & newName)
{
   _name = newName;
   return 0;
}

inline
std::string logFile::name()
{
   return _name;
}

inline
int logFile::chunkSize( hsize_t newChunkSize )
{
   _chunkSize = newChunkSize;
   return 0;
}

inline
hsize_t logFile::chunkSize()
{
   return _chunkSize;
}

inline
int logFile::maxSetSize( hsize_t newMaxSetSize )
{
   _maxSetSize = newMaxSetSize;
   return 0;
}

inline
hsize_t logFile::maxSetSize()
{
   return _maxSetSize;
}

inline
int logFile::maxFileSize( hsize_t newMaxFileSize )
{
   _maxFileSize = newMaxFileSize;
   return 0;
}

inline
hsize_t logFile::maxFileSize()
{
   return _maxFileSize;
}   

inline
int logFile::writeLog( std::shared_ptr<char> & data, hsize_t N, time::timespecX & ts)
{
   int status;

   //Check if we need a new file
   if(_currFileSize + N > _maxFileSize || _file_H == 0)
   {
      createFile(ts);
   }

   //Check if we need a new dataset
   if(_currSetSize + N > _maxSetSize || _dataSet_H == 0)
   {
      createSet(ts);
   }
   
   //hsize_t setSize = N;
   
   //Extend the current data set
   hsize_t newSize = _currSetSize + N;
    
   status = H5Dset_extent(_dataSet_H, &newSize);
   
   //Select a hyperslab.
   H5Handle_S filespace;
   filespace = H5Dget_space(_dataSet_H);
   
   hsize_t offset = _currSetSize;
  
   status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &N, NULL);
 
   H5Handle_S dataSpace_H;
   dataSpace_H = createSpace(N);
   status = H5Dwrite(_dataSet_H, H5T_NATIVE_CHAR, dataSpace_H, filespace, H5P_DEFAULT, data.get());
   
   _currSetSize = newSize;
   _currFileSize += N;
   
   return 0;
}

inline
int logFile::closeAll()
{
   _file_H.close();
   _dataSet_H.close();
   
   
   return 0;
}

   
inline
int logFile::createFile(time::timespecX & ts)
{
   std::string tstamp;
   getTimeStamp(tstamp, ts);

   //Create the standard log name
   std::string fname = _path + "/" + _name + "_" + tstamp + "." + _ext;

   //Create and open the file.
   //Note the handle assignment will close the previous file if open.
   _file_H = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   
   if(_file_H <= 0)
   {
      std::cerr << "logFile::createFile: Error by H5Fcreate.  At: " << __FILE__ << " " << __LINE__ << "\n";
      return -1;
   }
   
   //Reset counters.
   _currFileSize = 0;
   _currSet = 0;
   
   //Close the dataset if open.
   _dataSet_H.close();
      
   return 0;
}

inline
hid_t logFile::createSpace(hsize_t N)
{
   
   hsize_t maxDim = H5S_UNLIMITED;

   return H5Screate_simple(1, &N, &maxDim);
   
}

inline
int logFile::createSet(time::timespecX & ts)
{
   int status;
   
   if(_dataSet_H)
   {
      _dataSet_H.close();
      ++_currSet;
   }
   
   std::string setName = "set" + mx::convertToString(_currSet);
   
   H5Handle_P   cparms;
   hsize_t  chunk_dims = _chunkSize;
 
   cparms = H5Pcreate(H5P_DATASET_CREATE);
   
   status = H5Pset_chunk( cparms, 1, &chunk_dims);
   //status = H5Pset_deflate(cparms, 9); //This makes it worse!!!
   
   H5Handle_S dataSpace_H;
   dataSpace_H = createSpace(0);
   
   _dataSet_H = H5Dcreate2(_file_H, setName.c_str(), H5T_NATIVE_CHAR, dataSpace_H, H5P_DEFAULT, cparms, H5P_DEFAULT);
   _currSetSize = 0;
   
   H5Handle_S   attr_space_sec;
   H5Handle_A   attr_sec;
   
   H5Handle_S   attr_space_nsec;
   H5Handle_A   attr_nsec;
   
   attr_space_sec  = H5Screate(H5S_SCALAR);
   attr_sec = H5Acreate2(_dataSet_H, "Start sec", H5T_NATIVE_INT64, attr_space_sec, H5P_DEFAULT, H5P_DEFAULT);
   status = H5Awrite(attr_sec, H5T_STD_I64LE, &ts.time_s);
   
   attr_space_nsec  = H5Screate(H5S_SCALAR);
   attr_nsec = H5Acreate2(_dataSet_H, "Start nsec", H5T_NATIVE_INT32, attr_space_nsec, H5P_DEFAULT, H5P_DEFAULT);
   status = H5Awrite(attr_nsec, H5T_STD_I32LE, &ts.time_ns);
   
   return 0;
}
   

} //namespace logger 
} //namespace MagAOX 

#endif //logger_logFile_hpp
