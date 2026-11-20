#!/bin/bash
# Build the three indiserver variants compared in ../indiserver-chained-loop.md into ./build.
#   dev    : threaded indiserver on dev
#   devfix : dev plus keep-name-in-generic-getProperties.patch (the analysis-stage fix)
#   branch : jrmales/indiserver-getproperties-loop, the implemented fixes
#   stefan : stefan_schilperoort/dev-resurrector (077c0eb2 + 895e3cb8)
#   oldres : indilib-lineage resurrector indiserver (077c0eb2^ with dac13a5f's indiserver.c)
# Set REPO to the MagAOX checkout if this directory is not inside it.
S=$(cd $(dirname $0); pwd); R=${REPO:-$(git -C $S rev-parse --show-toplevel)}
b()
{
    local v=$1 ref=$2 isref=$3 srcs="indiserver.c fq.c"
    rm -rf $S/build/$v; mkdir -p $S/build/$v
    git -C $R archive $ref INDI/INDI INDI/liblilxml | tar -x -C $S/build/$v
    [ -n "$isref" ] && git -C $R show $isref:INDI/INDI/indiserver.c > $S/build/$v/INDI/INDI/indiserver.c
    [ $v = devfix ] && patch -s -p1 -d $S/build/$v < $S/keep-name-in-generic-getProperties.patch
    (cd $S/build/$v/INDI/liblilxml && gcc -O0 -g -c lilxml.c base64.c && ar rcs liblilxml.a lilxml.o base64.o)
    [ $v = oldres ] && srcs="indiserver.c strcat_varargs.c open_named_fifo.c fq.c"
    (cd $S/build/$v/INDI/INDI && gcc -O0 -g -DGIT_TAG_STRING='"sim"' -o indiserver -I../liblilxml $srcs ../liblilxml/liblilxml.a -lpthread -lz)
}
b dev dev
b devfix dev
b branch jrmales/indiserver-getproperties-loop
b stefan origin/stefan_schilperoort/dev-resurrector
b oldres 077c0eb2^ dac13a5f
