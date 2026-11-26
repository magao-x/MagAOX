#include "MCP3208.h"
#include <cstdint>
#include <fcntl.h>
#include <linux/spi/spidev.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

namespace MCP3208Lib {

namespace {

// Control bits for MCP3208 (12-bit ADC): the first byte holds 3 leading zeros and the start bit, the second the
// sgl/diff (mode) bit, the channel selection and 4 "don't care" bits; the third is padding.
void command(const std::uint8_t channel, const Mode m, std::uint8_t (&txData)[3]) {
    txData[0] = 0b00000110;
    txData[1] = (static_cast<std::uint8_t>(m) << 7) | static_cast<std::uint8_t>((channel & 0b00000111) << 4);
    txData[2] = 0b00000000;
}

// The result is a 12-bit number: the first 4 bits from the second byte and 8 bits from the third byte.
unsigned short decode(const std::uint8_t (&rxData)[3]) {
    return
        ((static_cast<unsigned short>(rxData[1]) & 0x0F) << 8) |
        (static_cast<unsigned short>(rxData[2]) & 0xFF);
}

}

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

    // The kernel spidev device, so several conversions can share one SPI message (one system call).
    const std::string path = "/dev/spidev" + std::to_string(this->_dev) + "." + std::to_string(this->_channel);
    const int handle = ::open(path.c_str(), O_RDWR);

    if (handle < 0) {
        throw std::runtime_error("failed to connect spi device " + path);
    }

    std::uint8_t mode = static_cast<std::uint8_t>(this->_flags & 0b11);
    std::uint8_t bits = 8;
    std::uint32_t speed = static_cast<std::uint32_t>(this->_baud);

    if (::ioctl(handle, SPI_IOC_WR_MODE, &mode) < 0 || ::ioctl(handle, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0 ||
        ::ioctl(handle, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
        ::close(handle);
        throw std::runtime_error("failed to configure spi device " + path);
    }

    this->_handle = handle;
}

void MCP3208::disconnect() {
    if (this->_handle < 0) {
        return;
    }

    if (::close(this->_handle) != 0) {
        throw std::runtime_error("failed to disconnect spi device");
    }

    this->_handle = -1;
}

unsigned short MCP3208::read(const std::uint8_t channel, const Mode m) const {
    unsigned short value = 0;
    this->read(&channel, &value, 1, m);
    return value;
}

void MCP3208::read(const std::uint8_t *channels, unsigned short *values, const int count, const Mode m) const {

    if (count < 1 || count > MAX_CHANNELS) {
        throw std::runtime_error("invalid MCP3208 channel count");
    }

    std::uint8_t txData[MAX_CHANNELS][3];
    std::uint8_t rxData[MAX_CHANNELS][3]{};
    spi_ioc_transfer transfers[MAX_CHANNELS]{};

    for (int i = 0; i < count; ++i) {
        command(channels[i], m, txData[i]);
        transfers[i].tx_buf = reinterpret_cast<std::uintptr_t>(txData[i]);
        transfers[i].rx_buf = reinterpret_cast<std::uintptr_t>(rxData[i]);
        transfers[i].len = 3;
        transfers[i].speed_hz = static_cast<std::uint32_t>(this->_baud);
        transfers[i].bits_per_word = 8;
        // Each conversion ends with chip select raised, as for separate transfers.
        transfers[i].cs_change = i + 1 < count;
    }

    const unsigned long request = _IOC(_IOC_WRITE, SPI_IOC_MAGIC, 0, count * sizeof(spi_ioc_transfer));

    if (::ioctl(this->_handle, request, transfers) < 0) {
        throw std::runtime_error("SPI transfer failed");
    }

    for (int i = 0; i < count; ++i) {
        values[i] = decode(rxData[i]);
    }
}

};  // End of namespace MCP3208Lib
