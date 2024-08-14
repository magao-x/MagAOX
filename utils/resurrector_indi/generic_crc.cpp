#include <iostream>
#include "generic_crc.hpp"
#include "string.h"

int
main(int argc, char** argv)
{
    int poly{0xa001};
    int width{16};
    int next_arg{1};
    bool ones_arg{true};
    bool dump_table{false};
    bool do_xorout{false};
    while (next_arg < argc)
    {
        if (!strncmp(argv[next_arg],"--poly=0x",9))
        {
            if (!sscanf(argv[next_arg]+9,"%x",&poly)) { break; };
            ++next_arg;
            continue;
        }
        if (!strncmp(argv[next_arg],"--width=",8))
        {
            if (!sscanf(argv[next_arg]+8,"%d",&width)) { break; };
            ++next_arg;
            continue;
        }
        if (!strcmp(argv[next_arg],"--zeros"))
        {
            ones_arg = false;
            ++next_arg;
            continue;
        }
        if (!strcmp(argv[next_arg],"--dump-table"))
        {
            dump_table = true;
            ++next_arg;
            continue;
        }
        if (!strcmp(argv[next_arg],"--xorout"))
        {
            do_xorout = true;
            ++next_arg;
            continue;
        }
        break;
    }
    CRC crc(poly, width, ones_arg, do_xorout);
    std::cerr << crc << '\n';
    if (dump_table) { crc.dump_table(std::cerr); }
    while (next_arg < argc)
    {
        std::cerr
        << std::hex
        << std::setw(width>>2)
        << std::setfill('0')
        << crc.crc_calc((unsigned char*) argv[next_arg]
                       ,strlen(argv[next_arg])
                       )
        << '\n';
        ++next_arg;
    }
  
    return 0;
}
