#ifndef MCP3208_H_AA185758_F169_4B8A_8158_6E4588F5B55F
#define MCP3208_H_AA185758_F169_4B8A_8158_6E4588F5B55F

#include <cstdint>
#include <linux/spi/spidev.h>

namespace MCP3208Lib {

enum class Mode : std::uint8_t {
    SINGLE = 1,
    DIFFERENTIAL = 0
};

class MCP3208 {
public:
    static const int DEFAULT_SPI_DEV = 0;
    static const int DEFAULT_SPI_CHANNEL = 0;
    static const int SPI_5V_BAUD = 3600000;
    static const int SPI_2_7V_BAUD = 1350000;
    static const int DEFAULT_SPI_BAUD = SPI_2_7V_BAUD;

    //https://www.analog.com/en/analog-dialogue/articles/introduction-to-spi-interface.html
    static const int DEFAULT_SPI_FLAGS = SPI_MODE_0;

    MCP3208(
        const int dev = DEFAULT_SPI_DEV,
        const int channel = DEFAULT_SPI_CHANNEL,
        const int baud = DEFAULT_SPI_BAUD,
        const int flags  = DEFAULT_SPI_FLAGS) noexcept;

    virtual ~MCP3208();

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