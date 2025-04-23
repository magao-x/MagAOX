// MIT License
//
// Copyright (c) 2021 Daniel Robertson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef MCP3008_H_AA185758_F169_4B8A_8158_6E4588F5B55F
#define MCP3008_H_AA185758_F169_4B8A_8158_6E4588F5B55F

#include <cstdint>
#include <linux/spi/spidev.h>

/**
 * Datasheet
 * https://cdn-shop.adafruit.com/datasheets/MCP3008.pdf
 */

namespace MCP3008Lib {

enum class Mode : std::uint8_t {
    SINGLE = 1,
    DIFFERENTIAL = 0
};

class MCP3008 {
public:
    static const int DEFAULT_SPI_DEV = 0;
    static const int DEFAULT_SPI_CHANNEL = 0;
    static const int SPI_5V_BAUD = 3600000;
    static const int SPI_2_7V_BAUD = 1350000;
    static const int DEFAULT_SPI_BAUD = SPI_2_7V_BAUD;

    //https://www.analog.com/en/analog-dialogue/articles/introduction-to-spi-interface.html
    static const int DEFAULT_SPI_FLAGS = SPI_MODE_0;

    MCP3008(
        const int dev = DEFAULT_SPI_DEV,
        const int channel = DEFAULT_SPI_CHANNEL,
        const int baud = DEFAULT_SPI_BAUD,
        const int flags  = DEFAULT_SPI_FLAGS) noexcept;
    
    virtual ~MCP3008();

    void connect();
    void disconnect();
    unsigned short read(const std::uint8_t channel, const Mode m = Mode::SINGLE) const;


protected:
    int _handle;
    int _dev;
    int _channel;
    int _baud;
    int _flags;

};

};
#endif