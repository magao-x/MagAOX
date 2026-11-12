/** \file ImageStreamIO.h
 * \brief Offline ADC reader API double; never opens shared memory, devices or native semaphores.
 */
#ifndef DDSPC_TEST_IMAGESTREAMIO_H
#define DDSPC_TEST_IMAGESTREAMIO_H
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <semaphore.h>
#include <sys/types.h>

/// \cond ADC_TRANSPORT_DOUBLE
#define _DATATYPE_UINT16 3
#define CIRCULAR_BUFFER 0x00001
#define ZAXIS_TEMPORAL 0x10000

/// Minimal metadata fields exercised by the actual native reader translation unit.
struct IMAGE_METADATA
{
    std::uint32_t naxis{ 3 }, size[3]{ 2, 1, 1 };
    std::uint8_t  datatype{ _DATATYPE_UINT16 };
    int           location{ -1 }, sem{ 1 }, write{ 0 };
    std::uint64_t imagetype{ CIRCULAR_BUFFER | ZAXIS_TEMPORAL }, cnt0{ 0 }, cnt1{ 0 };
    timespec      atime{}, writetime{};
};

/// Minimal borrowed image; storage belongs to a private test fixture.
struct IMAGE
{
    IMAGE_METADATA *md{ nullptr };
    struct
    {
        void *raw{ nullptr };
    } array;
    timespec      *atimearray{ nullptr }, *writetimearray{ nullptr };
    std::uint64_t *cntarray{ nullptr };
    sem_t        **semptr{ nullptr };
    pid_t         *semReadPID{ nullptr };
};

/// Resolve only the process-private fixture directory, never ImageStreamIO's runtime namespace.
int ImageStreamIO_filename( char       *path /**< [out] Destination. */,
                            std::size_t size /**< [in] Bound. */,
                            const char *name /**< [in] Fixture identity. */ );

/// Borrow a registered offline fixture; no OS stream access.
int ImageStreamIO_openIm( IMAGE *image /**< [out] Borrowed image. */, const char *name /**< [in] Fixture name. */ );

/// Return the fixture's sole reader slot.
int ImageStreamIO_getsemwaitindex( IMAGE *image /**< [in,out] Reader bookkeeping. */,
                                   int    hint /**< [in] Ignored hint. */ );

/// Ignore notification flushing; fake payloads do not use real semaphores.
void ImageStreamIO_semflush( IMAGE *image /**< [in] Ignored mapping. */, int index /**< [in] Ignored index. */ );

/// Release a fixture view without destroying its storage.
void ImageStreamIO_closeIm( IMAGE *image /**< [in,out] Borrowed view. */ );

/// Simulate an empty notification queue without invoking sem_trywait on fake storage.
int ddspc_adc_sem_trywait( sem_t *sem /**< [in] Ignored fake semaphore. */ );
#define sem_trywait ddspc_adc_sem_trywait
/// \endcond
#endif // DDSPC_TEST_IMAGESTREAMIO_H
