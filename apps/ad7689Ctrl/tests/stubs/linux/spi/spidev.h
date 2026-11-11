/** \file spidev.h
 * \brief Minimal Linux-compatible SPI UAPI for portable offline tests only.
 * \cond AD7689_TEST_STUBS
 */
#ifndef ad7689Ctrl_test_spidev_h
#define ad7689Ctrl_test_spidev_h
#include <cstdint>
#include <sys/ioctl.h>
#define SPI_MODE_0 0
#define SPI_NO_CS 0x40
struct spi_ioc_transfer
{
    uint64_t tx_buf;
    uint64_t rx_buf;
    uint32_t len;
    uint32_t speed_hz;
    uint16_t delay_usecs;
    uint8_t  bits_per_word, cs_change, tx_nbits, rx_nbits, word_delay_usecs, pad;
};
#define SPI_IOC_MESSAGE( N ) _IOW( 'k', 0, char[( N ) * sizeof( spi_ioc_transfer )] )
#define SPI_IOC_RD_LSB_FIRST _IOR( 'k', 2, uint8_t )
#define SPI_IOC_WR_LSB_FIRST _IOW( 'k', 2, uint8_t )
#define SPI_IOC_RD_BITS_PER_WORD _IOR( 'k', 3, uint8_t )
#define SPI_IOC_WR_BITS_PER_WORD _IOW( 'k', 3, uint8_t )
#define SPI_IOC_WR_MAX_SPEED_HZ _IOW( 'k', 4, uint32_t )
#define SPI_IOC_RD_MODE32 _IOR( 'k', 5, uint32_t )
#define SPI_IOC_WR_MODE32 _IOW( 'k', 5, uint32_t )
#endif
/** \endcond */
