#include "MCP3208.h"
#include <cstdint>
#include "lgpio.h"
#include <stdexcept>

namespace MCP3208Lib {

MCP3208::MCP3208(
    const int dev,
    const int channel,
    const int baud,
    const int flags) noexcept :
        _handle(-1),
        _dev(dev),
        _channel(channel),
        _baud(baud),
        _flags(flags) {
}

MCP3208::~MCP3208() {

    try {
        this->disconnect();
    }
    catch(...) {
        // Prevent propagation
    }

}

void MCP3208::connect() {

    if (this->_handle >= 0) {
        return;
    }

    const auto handle = ::lgSpiOpen(
        this->_dev,
        this->_channel,
        this->_baud,
        this->_flags);

    if (handle < 0) {
        throw std::runtime_error("failed to connect spi device");
    }

    this->_handle = handle;
}

void MCP3208::disconnect() {
    if (this->_handle < 0) {
        return;
    }

    if (::lgSpiClose(this->_handle) != 0) {
        throw std::runtime_error("failed to disconnect spi device");
    }

    this->_handle = -1;
}

unsigned short MCP3208::read(const std::uint8_t channel, const Mode m) const {

    // Control bits for MCP3208 (12-bit ADC)
    // First bit is single or differential mode
    // Next three bits are channel selection (0 to 7)
    // The last four bits are padding
    const std::uint8_t ctrl =
        (static_cast<std::uint8_t>(m) << 7) |
        static_cast<std::uint8_t>((channel & 0b00000111) << 4)
        ;

    const std::uint8_t byteCount = 3;

    // MCP3208 uses a 3-byte message (start bit + control + padding)
    const std::uint8_t txData[byteCount] = {
        0b00000110, // 3 leading zeros and start bit for MCP3208
        ctrl,       // sgl/diff (mode), d2, d1, d0, 4x "don't care" bits
        0b00000000  // Padding bits (don't care)
    };

    std::uint8_t rxData[byteCount]{0};

    const auto bytesTransferred = ::lgSpiXfer(
        this->_handle,
        reinterpret_cast<const char*>(txData),
        reinterpret_cast<char*>(rxData), byteCount);

    if (bytesTransferred != byteCount) {
        throw std::runtime_error("SPI transfer failed");
    }

    // For MCP3208, the result is a 12-bit number:
    // First 4 bits from the second byte and 8 bits from the third byte.
    return
        ((static_cast<unsigned short>(rxData[1]) & 0x0F) << 8) | // Top 4 bits from second byte
        (static_cast<unsigned short>(rxData[2]) & 0xFF);         // Bottom 8 bits from third byte
}

};  // End of namespace MCP3208Lib