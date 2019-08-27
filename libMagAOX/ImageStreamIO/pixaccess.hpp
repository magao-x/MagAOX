

///\todo document this file

#ifndef pixaccess_h
#define pixaccess_h

#include "ImageStruct.hpp"

///Function to cast the data type to float.
/** Accesses the imdata pointer at linear position idx, and then casts the result to float.
  *
  * \returns the value of the image at linear position idx cast to float
  * 
  * \tparam dataT the type of the image data.
  */ 
template<typename returnT, typename dataT>
returnT getPix( void *imdata, ///< [in] Pointer to the image data 
                size_t idx    ///< [in] Linear position of the pixel in the imdata
              )
{ 
   return (returnT) ((dataT*) imdata)[idx];
}

///Get the function pointer for getPix for the type
template<typename returnT, int imageStructDataT>
returnT (*getPixPointer())(void*, size_t)
{
   return &getPix<returnT, typename imageStructDataType<imageStructDataT>::type>;
}

template<typename returnT>
returnT (*getPixPointer(int imageStructDataT))(void*, size_t) 
{
   switch(imageStructDataT)
   {
      case IMAGESTRUCT_UINT8:
         return getPixPointer<returnT, IMAGESTRUCT_UINT8>();
      case IMAGESTRUCT_INT8:
         return getPixPointer<returnT, IMAGESTRUCT_INT8>();
      case IMAGESTRUCT_UINT16:
         return getPixPointer<returnT, IMAGESTRUCT_UINT16>();
         break;
      case IMAGESTRUCT_INT16:
         return getPixPointer<returnT, IMAGESTRUCT_INT16>();
      case IMAGESTRUCT_UINT32:
         return getPixPointer<returnT, IMAGESTRUCT_UINT32>();
      case IMAGESTRUCT_INT32:
         return getPixPointer<returnT, IMAGESTRUCT_INT32>();
      case IMAGESTRUCT_UINT64:
         return getPixPointer<returnT, IMAGESTRUCT_UINT64>();
      case IMAGESTRUCT_INT64:
         return getPixPointer<returnT, IMAGESTRUCT_INT64>();
      case IMAGESTRUCT_FLOAT:
         return getPixPointer<returnT, IMAGESTRUCT_FLOAT>();
      case IMAGESTRUCT_DOUBLE:
         return getPixPointer<returnT, IMAGESTRUCT_DOUBLE>();
      default:
         std::cerr << "getPixPointer: Unknown or unsupported data type. " << __FILE__ << " " << __LINE__ << "\n";
         return nullptr;
   }
}
#endif



