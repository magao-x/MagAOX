/** \file streamWriterSizing_test.cpp
 * \brief Catch2 tests for streamWriter buffer sizing helpers.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup streamWriter_files
 */

#include "../../../tests/testXWC.hpp"

#include "../streamWriter.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup streamWriter_unit_test streamWriter Unit Tests
 * \brief Unit tests for the streamWriter application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `streamWriter` unit tests.
/** \ingroup streamWriter_unit_test
 */
namespace streamWriterTest
{

/// Verify `streamWriter::getCircBuffLengths()` selects bounded circular-buffer and write-chunk sizes.
/**
 * \ingroup streamWriter_unit_test
 */
SCENARIO( "streamWriter Buffer Sizing", "[streamWriter]" )
{
    // clang-format off
    #ifdef STREAMWRITER_TEST_DOXYGEN_REF
    streamWriter::getCircBuffLengths( *(size_t *)nullptr,
                                      *(double *)nullptr,
                                      *(size_t *)nullptr,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0 );
    #endif
    // clang-format on

    GIVEN( "A default constructed streamWriter" )
    {
        WHEN( "default configurations" )
        {
            size_t maxCircBuffLength   = 1024;
            double maxCircBuffSize     = 2048.0;
            size_t maxWriteChunkLength = 512;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 1024;
            uint32_t height   = 1024;
            size_t   typeSize = 2;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 1024 );
            REQUIRE( circBuffSize == 2048 );
            REQUIRE( writeChunkLength == 512 );
        }

        WHEN( "larger frame size using all of max size" )
        {
            size_t maxCircBuffLength   = 1024;
            double maxCircBuffSize     = 2048.0;
            size_t maxWriteChunkLength = 512;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 2048;
            uint32_t height   = 2048;
            size_t   typeSize = 2;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 256 );
            REQUIRE( circBuffSize == 2048 );
            REQUIRE( writeChunkLength == 128 );
        }

        WHEN( "largest frame size possible using all of max size" )
        {
            size_t maxCircBuffLength   = 1024;
            double maxCircBuffSize     = 1024.0;
            size_t maxWriteChunkLength = 512;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 16384;
            uint32_t height   = 16384;
            size_t   typeSize = 2;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 2 );
            REQUIRE( circBuffSize == 1024 );
            REQUIRE( writeChunkLength == 1 );
        }

        WHEN( "Exceeding largest frame size possible" )
        {
            size_t maxCircBuffLength   = 1024;
            double maxCircBuffSize     = 1024.0;
            size_t maxWriteChunkLength = 512;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 16385;
            uint32_t height   = 16385;
            size_t   typeSize = 2;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 0 );
            REQUIRE( circBuffSize == 0 );
            REQUIRE( writeChunkLength == 0 );
        }

        WHEN( "Exceeding largest frame size possible by a lot" )
        {
            size_t maxCircBuffLength   = 1024;
            double maxCircBuffSize     = 1024.0;
            size_t maxWriteChunkLength = 512;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 56385;
            uint32_t height   = 56385;
            size_t   typeSize = 2;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 0 );
            REQUIRE( circBuffSize == 0 );
            REQUIRE( writeChunkLength == 0 );
        }

        WHEN( "LOWFS-like setup, 32x32 frames" )
        {
            size_t maxCircBuffLength   = 524288;
            double maxCircBuffSize     = 1024.0;
            size_t maxWriteChunkLength = 16384;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 32;
            uint32_t height   = 32;
            size_t   typeSize = 2;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 524288 );
            REQUIRE_THAT( circBuffSize, Catch::Matchers::WithinAbs( 1024, 0.00001 ) );
            REQUIRE( writeChunkLength == 16384 );
            REQUIRE( ( circBuffLength % writeChunkLength ) == 0 );
        }

        WHEN( "LOWFS-like setup, 512x512 frames" )
        {
            size_t maxCircBuffLength   = 524288;
            double maxCircBuffSize     = 1024.0;
            size_t maxWriteChunkLength = 16384;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 512;
            uint32_t height   = 512;
            size_t   typeSize = 2;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 2048 );
            REQUIRE_THAT( circBuffSize, Catch::Matchers::WithinAbs( 1024, 0.00001 ) );
            REQUIRE( writeChunkLength == 64 );
            REQUIRE( ( circBuffLength % writeChunkLength ) == 0 );
        }

        WHEN( "LOWFS-like setup, 3200x3200 frames" )
        {
            size_t maxCircBuffLength   = 524288;
            double maxCircBuffSize     = 1024.0;
            size_t maxWriteChunkLength = 16384;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 3200;
            uint32_t height   = 3200;
            size_t   typeSize = 2;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 52 );
            REQUIRE_THAT( circBuffSize, Catch::Matchers::WithinAbs( 1015.625, 0.00001 ) );
            REQUIRE( writeChunkLength == 1 );
            REQUIRE( ( circBuffLength % writeChunkLength ) == 0 );
        }

        WHEN( "computed odd circular buffers round down and zero write chunks promote to one" )
        {
            size_t maxCircBuffLength   = 1000;
            double maxCircBuffSize     = 11.0 / 1048576.0;
            size_t maxWriteChunkLength = 1;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 1;
            uint32_t height   = 1;
            size_t   typeSize = 1;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 10 );
            REQUIRE_THAT( circBuffSize, Catch::Matchers::WithinAbs( 10.0 / 1048576.0, 1e-12 ) );
            REQUIRE( writeChunkLength == 1 );
        }

        WHEN( "computed write chunks shrink until they evenly divide the circular buffer" )
        {
            size_t maxCircBuffLength   = 10;
            double maxCircBuffSize     = 9.0 / 1048576.0;
            size_t maxWriteChunkLength = 4;

            size_t circBuffLength;
            double circBuffSize;
            size_t writeChunkLength;

            uint32_t width    = 1;
            uint32_t height   = 1;
            size_t   typeSize = 1;

            streamWriter::getCircBuffLengths( circBuffLength,
                                              circBuffSize,
                                              writeChunkLength,
                                              maxCircBuffLength,
                                              maxCircBuffSize,
                                              maxWriteChunkLength,
                                              width,
                                              height,
                                              typeSize );

            REQUIRE( circBuffLength == 8 );
            REQUIRE_THAT( circBuffSize, Catch::Matchers::WithinAbs( 8.0 / 1048576.0, 1e-12 ) );
            REQUIRE( writeChunkLength == 2 );
            REQUIRE( ( circBuffLength % writeChunkLength ) == 0 );
        }
    }
}

} // namespace streamWriterTest

} // namespace libXWCTest
