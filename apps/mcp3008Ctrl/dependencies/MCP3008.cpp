#include "MCP3008.h"
#include <cstdint>
#include <lgpio.h>
#include <stdexcept>

namespace MCP3008Lib {

MCP3008::MCP3008(
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

MCP3008::~MCP3008() {

    try {
        this->disconnect();
    }
    catch(...) {
        //prevent propagation
    }

}

void MCP3008::connect() {

    if(this->_handle >= 0) {
        return;
    }

    const auto handle = ::lgSpiOpen(
        this->_dev,
        this->_channel,
        this->_baud,
        this->_flags);

    if(handle < 0) {
        throw std::runtime_error("failed to connect spi device");
    }

    this->_handle = handle;

}

void MCP3008::disconnect() {

    if(this->_handle < 0) {
        return;
    }

    if(::lgSpiClose(this->_handle) != 0) {
        throw std::runtime_error("failed to disconnect spi device");
    }

    this->_handle = -1;

}

unsigned short MCP3008::read(const std::uint8_t channel, const Mode m) const {

    //control bits
    //first bit is single or differential mode
    //next three bits are channel selection
    //last four bits are ignored
    const std::uint8_t ctrl =
        (static_cast<std::uint8_t>(m) << 7) |
         static_cast<std::uint8_t>((channel & 0b00000111) << 4)
        ;

    const std::uint8_t byteCount = 3;

    const std::uint8_t txData[byteCount] = {
        0b00000001, //seven leading zeros and start bit
        ctrl,       //sgl/diff (mode), d2, d1, d0, 4x "don't care" bits
        0b00000000  //8x "don't care" bits
        };

    std::uint8_t rxData[byteCount]{0};

    const auto bytesTransferred = ::lgSpiXfer(
        this->_handle,
        reinterpret_cast<const char*>(txData),
        reinterpret_cast<char*>(rxData),
        byteCount);

    if(bytesTransferred != byteCount) {
        throw std::runtime_error("spi transfer failed");
    }

    //first 14 bits are ignored
    //no need to AND with 0x3ff this way
    return
        ((static_cast<unsigned short>(rxData[1]) & 0b00000011) << 8) |
         (static_cast<unsigned short>(rxData[2]) & 0b11111111);

}

};