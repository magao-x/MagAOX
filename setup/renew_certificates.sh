#!/usr/bin/env bash
legoDataPath=/opt/lego
domain=exao1.magao-x.org
email=lynx@magao-x.org
lego --accept-tos \
    --email $email \
    --dns manual \
    --domains $domain \
    --path $legoDataPath \
    run
