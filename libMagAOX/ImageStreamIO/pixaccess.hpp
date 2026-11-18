

///\todo document this file

#ifndef pixaccess_h
#define pixaccess_h

#include "ImageStruct.hpp"
#include <iostream>

///Function to cast the data type to float.
/** Accesses the imdata pointer at linear position idx, and then casts the result to float.
 *
 * \returns the value of the image at linear position idx cast to float
 *
 * \tparam dataT the type of the image data.
 */
template <typename returnT, typename dataT>
returnT getPix( void  *imdata, ///< [in] Pointer to the image data
                size_t idx     ///< [in] Linear position of the pixel in the imdata
)
{
    return (returnT)( (dataT *)imdata )[idx];
}

///Get the function pointer for getPix for the type
template <typename returnT, int imageStructDataT>
returnT ( *getPixPointer() )( void *, size_t )
{
    return &getPix<returnT, typename imageStructDataType<imageStructDataT>::type>;
}

template <typename returnT>
returnT ( *getPixPointer( int imageStructDataT ) )( void *, size_t )
{
    switch( imageStructDataT )
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

/// Convert a frame of pixels, adding them to or replacing the destination
/** A loop over the whole frame with known types, which the compiler can vectorize, unlike calling getPix for each pixel
 * through a function pointer.
 *
 * \tparam returnT the type of the destination
 * \tparam dataT the type of the image data
 * \tparam add if true the pixels are added to the destination, otherwise they replace it
 */
template <typename returnT, typename dataT, bool add>
void getPixFrame( returnT    *dest,   /**< [in/out] the destination, npix long */
                  const void *imdata, /**< [in] pointer to the image data */
                  size_t      npix    /**< [in] the number of pixels */
)
{
    const dataT *src = static_cast<const dataT *>( imdata );

    for( size_t n = 0; n < npix; ++n )
    {
        if constexpr( add )
        {
            dest[n] += static_cast<returnT>( src[n] );
        }
        else
        {
            dest[n] = static_cast<returnT>( src[n] );
        }
    }
}

/// Get the function pointer for getPixFrame for the type
/**
 * \returns the function pointer, or nullptr if the type is not supported
 *
 * \tparam returnT the type of the destination
 * \tparam add if true the pixels are added to the destination, otherwise they replace it
 */
template <typename returnT, bool add>
void ( *getPixFramePointer( int imageStructDataT /**< [in] the ImageStreamIO type code */ ) )( returnT *,
                                                                                               const void *,
                                                                                               size_t )
{
    switch( imageStructDataT )
    {
    case IMAGESTRUCT_UINT8:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_UINT8>::type, add>;
    case IMAGESTRUCT_INT8:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_INT8>::type, add>;
    case IMAGESTRUCT_UINT16:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_UINT16>::type, add>;
    case IMAGESTRUCT_INT16:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_INT16>::type, add>;
    case IMAGESTRUCT_UINT32:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_UINT32>::type, add>;
    case IMAGESTRUCT_INT32:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_INT32>::type, add>;
    case IMAGESTRUCT_UINT64:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_UINT64>::type, add>;
    case IMAGESTRUCT_INT64:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_INT64>::type, add>;
    case IMAGESTRUCT_FLOAT:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_FLOAT>::type, add>;
    case IMAGESTRUCT_DOUBLE:
        return &getPixFrame<returnT, imageStructDataType<IMAGESTRUCT_DOUBLE>::type, add>;
    default:
        std::cerr << "getPixFramePointer: Unknown or unsupported data type. " << __FILE__ << " " << __LINE__ << "\n";
        return nullptr;
    }
}
#endif
