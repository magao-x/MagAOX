/*
 * Test Cyclic Redundancy Code robustness on
 * 9-hexadecimal-digit string of current time
 * using burst bit errors
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
class twobytes {
public:
    unsigned short I;
    unsigned char *pb2;
    twobytes(unsigned short Ival=1) : I(Ival)
    {
        fix_pb2();
    }
    void fix_pb2() { pb2 = (unsigned char*) &I; }
};

static std::vector<twobytes> v2b(0);

int
main(int argc, char** argv)
{
    assert(sizeof(unsigned short)==2);
    //int poly{0xe5};
    //int width{8};
    int poly{0x30};
    int width{6};
    //int poly{0x01};
    //int width{1};
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
    assert(width<9 && width>0);

    CRC thecrc(poly, width, ones_arg, do_xorout);

    std::cerr << thecrc << '\n';
    if (dump_table) { thecrc.dump_table(std::cerr); }

    for (int nbits=1; nbits<3; ++nbits)
    {
        unsigned short Ibase((((unsigned short)1)<<nbits)-1);
        unsigned short I(Ibase<<(15-nbits));
        while(I>=Ibase)
        {
            twobytes one2b(I);
            if (one2b.pb2[0]) { v2b.push_back(one2b); }
            I >>= 1;
        }
    }
    for (auto& it : v2b) { it.fix_pb2(); }
  
    time_t tt = time((time_t*)0);
    std::ostringstream oss(std::ostringstream::ate);
    int32_t xorcrc;
    int32_t hexcrc;
    unsigned long failures{0};
#   define ONEY 0x20000000
    for (time_t srem = ONEY; srem >= 0; --srem, ++tt)
    {
        oss.str("");
        oss << std::hex << std::uppercase << std::setw(9) << std::setfill('0') << tt;
        std::string hb(oss.str());
        size_t hbsize{hb.size()};
        size_t hbsizeplus1{hbsize+1};

        int32_t crc = thecrc.crc_calc((const unsigned char*)hb.c_str(),hbsize);

        oss.str("");
        oss << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << crc;

        hb += oss.str();
#       if TEST_CRC_DEBUG
        std::string hbsave{hb};
        std::cerr<<hb<<"<<<\n";
#       endif//TEST_CRC_DEBUG

        for (size_t ihb=0; ihb<=hbsizeplus1; ++ihb)
        {
            for (auto const& it: v2b)
            {
                bool not_detected(true);
                hb[ihb] ^= it.pb2[0];
                if (it.pb2[1] && ihb<hbsizeplus1) { hb[ihb+1] ^= it.pb2[1]; }

#               if TEST_CRC_DEBUG
                assert( hbsave!=hb);
                std::cerr
                <<(not_detected ? "true" : "false")<<':'
                ;
#               endif//TEST_CRC_DEBUG

                for (size_t ihex=ihb; ihex<=hbsizeplus1 && not_detected; ++ihex)
                {

#                   if TEST_CRC_DEBUG
                    std::cerr
                    <<(not_detected ? "true" : "false")<<"(ihex="<<ihex<<"before):"
                    ;
#                   endif//TEST_CRC_DEBUG

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

                if (it.pb2[1] && ihb<hbsizeplus1) { hb[ihb+1] ^= it.pb2[1]; }
                hb[ihb] ^= it.pb2[0];

#               if TEST_CRC_DEBUG
                assert( hbsave==hb);
#               endif//TEST_CRC_DEBUG

                if (!not_detected) { continue; }
                ++failures;
            }
        }
        //std::cerr<<hb<<':';

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
