/// Generic CRC class
/** Cf. http://www.ross.net/crc/download/crc_v3.txt
  * - Width:  up to 32-bits; set by highest bit of input polynomial
  * - Polynomial value:  int32_t input argument
  * - Init:  bool input argument; all zeros (false) or all ones (true)
  * - Reflected input
  *   - Bit 0 is first bit of each byte
  *   - Polynomial's implicit 1 is one bit to the right of bit 0
  * - No Reflected output
  *   - Reflecting polynomial achieves this, I think
  * - No XORing of output
  * - Incremental update of 
  */
#include <iomanip>
#include <ostream>
#include <sys/types.h>

class CRC {
private:
    int32_t poly{0xa001};
    int32_t width{16};
    int32_t poly_mask{(int32_t)0x0000ffff};
    int32_t init_crc_reg{0x0000ffff};
    int32_t last_crc_reg{0};
    int32_t table[256]{0};
public:

    /// Constructor
    /** Store polynomial
      * Populate table
      * Calculate mask
      * Assign initial value for CRC register
      */
    CRC(int32_t poly_arg, int32_t width_arg, bool ones)
    : poly(poly_arg), width(width_arg)
    {
        for (int32_t i=1; i<256; ++i)
        {
           table[i] = crc_algorithm_on_one_byte(0, i, 0);
        }
        poly_mask = (1<<width) - 1;
        last_crc_reg = init_crc_reg = ones ? poly_mask : 0;
    }

    /// Calculate CRC
    /** Restart with new input stream or incremental from last run
      */
    int32_t
    crc_calc(const unsigned char* p, unsigned int len, bool init = true)
    {
        if (init) { last_crc_reg = init_crc_reg; }
        while (len--)
        {
            int32_t index = (last_crc_reg ^ *(p++)) & 0x0ff;
            last_crc_reg >>= 8;
            last_crc_reg &= 0x00ffff;;
            last_crc_reg ^= table[index];
        }
        return last_crc_reg & poly_mask;
    }

    /// Calculate change for one input byte
    /** Used to calculate table entries in Constructor
      */
    int32_t
    crc_algorithm_on_one_byte(int32_t crcbits, int32_t newint32_t, int32_t rightshift)
    {
          crcbits ^= ((newint32_t >> rightshift) & 0x0ff);
          for (int32_t ibit=0; ibit<8; ++ibit)
          {
             crcbits = ((crcbits>>1)&0x7ffffff) ^ ((crcbits&1) ? poly : 0);
          }
          return crcbits;
    }

    friend std::ostream& operator<<(std::ostream& os, const CRC& crcobj);

};

/// Output information about CRC
std::ostream& operator<<(std::ostream& os, const CRC& crcobj)
{
    os
    << "CRC[poly=0x" << std::hex << std::setw(8) << std::setfill('0') << crcobj.poly
    << ";width=" << std::dec << crcobj.width
    << ";mask=0x" << std::hex << std::setw(8) << std::setfill('0') << crcobj.poly_mask
    << ";init_crc_reg=0x" << std::hex << std::setw(8) << std::setfill('0') << crcobj.init_crc_reg
    << "]";
    return os;
}
