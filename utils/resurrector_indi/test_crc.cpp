/*
 * Test Cyclic Redundancy Code (CRC) robustness
 * on 9-hexadecimal-digit string of time_t time
 * since the Epoch using burst bit errors
 */
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <assert.h>

#include "ctype.h"
#include "string.h"
#include "time.h"

#include "generic_crc.hpp"

/* Burst bit errors
 * I - 16-bit unsigned integer
 * pb2 - pointer to I as two 8-bit unsigned integers
 *       - these will contain bit error locations of
 *         several bits in a row, that will be XORed
 *         with the bits of two string characters
 */
class twobytes
{
public:
    unsigned short I;
    unsigned char *pb2;
    twobytes(unsigned short Ival=1) : I(Ival)
    {
        fix_pb2();
    }
    void fix_pb2() { pb2 = (unsigned char*) &I; }
};


int
main(int argc, char** argv)
{
    assert(sizeof(unsigned short)==2);

    // Initialize the CRC parameters; these may be changed via
    // command-line arguments in the future
    int poly{0xe5};
    int width{8};
    //int poly{0x30};
    //int width{6};
    //int poly{0x01};
    //int width{1};

    //Process command-line arguments
    int next_arg{1};
    bool ones_arg{false};
    bool dump_table{false};
    bool do_xorout{false};

    while (next_arg < argc)
    {
        if (!strcmp(argv[next_arg],"--dump-table"))
        {
            dump_table = true;
            ++next_arg;
            continue;
        }
        break;
    }
    // Set limits for CRC width in bits
    assert(width<9 && width>0);

    // Instantiate the CRC class
    CRC thecrc(poly, width, ones_arg, do_xorout);
    std::cerr << thecrc << '\n';
    if (dump_table) { thecrc.dump_table(std::cerr); }

    // Set up the bit errors as a vector of class twobytes instances
    std::vector<twobytes> v2b(0);
    for (int nbits=1; nbits<6; ++nbits)  // 1- to 5-bit burst errors
    {
        // Set the low nbits bits of Ibase to 1s
        unsigned short Ibase((((unsigned short)1)<<nbits)-1);
        // Move those low nbits bits to the high nbits bits of I
        unsigned short I(Ibase<<(15-nbits));
        while(I>=Ibase)
        {
            // Instantiate a twobytes using I
            twobytes one2b(I);
            // Push it onto vector IFF the 1st byte is non-zero
            if (one2b.pb2[0]) { v2b.push_back(one2b); }
            //v2b.back().fix_pb2();  // why does this not work?
            I >>= 1;
        }
    }
    // Fix the .pb2 pointers
    for (auto& it : v2b) { it.fix_pb2(); }

    // Counter of undetected burst bit errors; should remain 0
    unsigned long failures{0};

    // tt:  signed integer seconds since the Epoch
    // oss:  stringstream to contain hexadecimal chars
    // xorcrc:  CRC of 1st 9 hex characters after bit errors are applied
    // hexcrc:  CRC value read from last two hex chars, after bit errors
    time_t tt = time((time_t*)0);
    std::ostringstream oss(std::ostringstream::ate);
    int32_t xorcrc;
    int32_t hexcrc;
    // Loop over TESTCOUNT seconds from the current time (July, 2022)
    // - Decrement srem toward zero, increment tt, by one on each pass
    // - Current time is [0x60000000]s past epoch, so ~0x20000000s will
    //   surpass 32-bit rollover at 0x80000000 (August, 2038)
#   define TESTCOUNT 0x20000000
    for (time_t srem = TESTCOUNT; srem >= 0; --srem, ++tt)
    {
        // Write tt to stringstream as 9 uppercase hex characters with
        // leading 0s
        oss.str("");
        oss << std::hex << std::uppercase << std::setw(9) << std::setfill('0') << tt;
        // Copy result to hexbeat (hb) string
        // - Get length, which will be position of first of 2 CRC chars,
        //   and length+1, which will be position of last of 2 CRC chars
        std::string hb(oss.str());
        size_t hbsize{hb.size()};
        size_t hbsizeplus1{hbsize+1};

        // Calculate CRC of 9-char hexbeat, convert to 2 uppercase hex
        // characters, append to hb
        int32_t crc = thecrc.crc_calc((const unsigned char*)hb.c_str(),hbsize);
        oss.str("");
        oss << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << crc;
        hb += oss.str();

#       if TEST_CRC_DEBUG
        std::string hbsave{hb};
        std::cerr<<hb<<"<<<\n";
#       endif//TEST_CRC_DEBUG

        // Loop over positions of characters in [hb+CRC] string
        for (size_t ihb=0; ihb<=hbsizeplus1; ++ihb)
        {
            // Loop over burst bit errors (vector of twobytes instances)
            for (auto const& it: v2b)
            {
                // Assume errors are not detected until they are
                bool not_detected(true);
                // Apply burst errors to char at current position
                hb[ihb] ^= it.pb2[0];
                // Apply burst errors to next char IFF BOTH errors are
                // non-zero AND current position is not last of string
                if (it.pb2[1] && ihb<hbsizeplus1) { hb[ihb+1] ^= it.pb2[1]; }

#               if TEST_CRC_DEBUG
                assert( hbsave!=hb);
                std::cerr
                <<(not_detected ? "true" : "false")<<':'
                ;
#               endif//TEST_CRC_DEBUG

                // Loop over remaining positions in string
                for (size_t ihex=ihb; ihex<=hbsizeplus1 && not_detected; ++ihex)
                {

#                   if TEST_CRC_DEBUG
                    std::cerr
                    <<(not_detected ? "true" : "false")<<"(ihex="<<ihex<<"before):"
                    ;
#                   endif//TEST_CRC_DEBUG

                    // Assume error will be detected if any char is
                    // EITHER a hexadecimal digit OR a lowercase char
                    not_detected = not_detected && isxdigit(hb[ihex]);
                    not_detected = not_detected && !islower(hb[ihex]);

#                   if TEST_CRC_DEBUG
                    if (not_detected) { continue; }
                    std::cerr
                    <<ihb<<':'
                    <<ihex<<':'
                    <<(0x0ff&(int)hb[ihex])<<':'
                    <<(isxdigit(hb[ihex]) ? "true" : "false")<<':'
                    <<(!islower(hb[ihex]) ? "true" : "false")<<':'
                    <<(not_detected ? "true" : "false")<<':'
                    ;
#                   endif//TEST_CRC_DEBUG

                }

                // If no error was detected above, calculate the CRC of
                // the first 9 chars of the error-applied string, and
                // compare that CRC to the last 2 chars of the string
                if (not_detected)
                {

#                   if TEST_CRC_DEBUG
                    std::cerr<<hb<<"+\n";
#                   endif//TEST_CRC_DEBUG

                    xorcrc = thecrc.crc_calc((const unsigned char*)hb.c_str(),hbsize);
                    assert(1==sscanf(hb.substr(hbsize).c_str(),"%x", &hexcrc));
                    not_detected &= (xorcrc == hexcrc);
                }

#               if TEST_CRC_DEBUG
                else
                {
                    std::cerr<<hb<<"-\n";
                }
#               endif//TEST_CRC_DEBUG

                // Back out burst errors
                if (it.pb2[1] && ihb<hbsizeplus1) { hb[ihb+1] ^= it.pb2[1]; }
                hb[ihb] ^= it.pb2[0];

#               if TEST_CRC_DEBUG
                assert( hbsave==hb);
#               endif//TEST_CRC_DEBUG

                // Increment undetected error counter
                if (not_detected) { ++failures; }
            }
        }

        // Print counter approximately once per year (32Ms) of tt values
        if (srem&0x01ffffff) { continue; }

        std::cerr
        << tt
        << ':' << hb
        << ':' << srem
        << ':' << failures
        << ':' << (ctime(&tt) ? ctime(&tt) : "?\n")
        ;
    }

    return failures!=0;
}
