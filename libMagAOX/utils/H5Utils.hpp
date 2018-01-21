/** \file H5Utils.hpp 
  * \brief Utilities for HDF5.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-08-28 created by JRM
  */ 

#ifndef utils_H5Utils_hpp
#define utils_H5Utils_hpp

#include <hdf5.h>

namespace MagAOX
{
namespace utils 
{

struct H5FileT
{
   static herr_t close( hid_t & h )
   {
      return H5Fclose(h);
   }
};

struct H5DatasetT
{
   static herr_t close( hid_t & h )
   {
      return H5Dclose(h);
   }
};

struct H5DataspaceT
{
   static herr_t close( hid_t & h )
   {
      return H5Sclose(h);
   }
};

struct H5PropertyT
{
   static herr_t close( hid_t & h )
   {
      return H5Pclose(h);
   }
};

struct H5AttributeT
{
   static herr_t close( hid_t & h )
   {
      return H5Aclose(h);
   }
};

///A somewhat smart HDF5 handle.
/** Makes sure that the associated hdf5 library resources are closed when out of scope.
  * Does not do reference counting, so copy and assignment are deleted. Assignment operator from hid_t is the only way to
  * set the handle, and the hid_t conversion operator allows this to be passed directly to hdf5 library functions.
  *
  * \note You must never call the raw hdf5 close function (e.g. H5Fclose) on one of these handles.
  * 
  * \tparam T is one of the handle types.
  */ 
template<class T>
class H5Handle 
{
protected:
   hid_t _hand {0}; ///< The underlying handle.

public: 
   //Have to define since we're deleting the copy constructor apparently.
   H5Handle(){;}
   
   
   //Prevent copying, which would risk deleting the handle without all copies knowing about it.
   H5Handle(const H5Handle &) = delete; 
   H5Handle & operator=(const H5Handle &) = delete;

   
   
   ///Assignment from a basic hdf5 handle.
   /** Note that if the handle has already been assigned, then it is closed first.
     *
     */
   hid_t & operator=(const hid_t & hand /**< [in] the basic handle*/)
   {
      if(_hand) close();
      
      _hand = hand;
      
      return _hand; ///\returns a reference to the handle
   }
   
   ///Conversion operator
   operator hid_t()
   {
      return _hand;
   }
   
   ///Close the handle
   /** Calls the close function of the handle type T
     */
   herr_t close()
   {      
      if(_hand == 0) return 0;
      
      herr_t rv = T::close(_hand); /// \returns the herr_t code from the hdf5 close function
      
      _hand = 0;
      
      return rv;
   }
   
   ///Destructor.  Calls the close function.
   ~H5Handle()
   {
      close();
   }
};

///Handle for an HDF5 file.
typedef H5Handle<H5FileT> H5Handle_F;

///Handle for an HDF5 dataset.
typedef H5Handle<H5DatasetT> H5Handle_D;

///Handle for an HDF5 dataspace.
typedef H5Handle<H5DataspaceT> H5Handle_S;

///Handle for an HDF5 property list.
typedef H5Handle<H5PropertyT> H5Handle_P;

///Handle for an HDF5 attribute.
typedef H5Handle<H5AttributeT> H5Handle_A;

} //namespace utils
} //namespace MagAOX

#endif //utils_H5Utils_hpp

