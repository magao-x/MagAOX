
#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../streamWriter.hpp"

using namespace MagAOX::app;

using namespace MagAOX::app;

SCENARIO( "streamWriter Buffer Sizing", "[streamWriter]" )
{
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
    }
}
