#!/bin/bash

# ./scripts/import-nnom-net.sh ../audio/implement/nnom/ ../audio/train/.cache/kws_nnom/weights.h kws_nnom src/ai/nnom/

if [ $# -ne 4 ]
then
  echo "Usage: "
  echo "  ./scripts/import-nnom-net.sh <nnom repo dir> <weights.h> <net name> <destination for src>"
  exit
fi

NNOM_SRC=$1
WEIGHTS=$2
NET_NAME=$3
DST=$4

echo "Copy NNoM sources from $NNOM_SRC"
mkdir -p $DST
cp -rv $NNOM_SRC/inc $DST/
cp -rv $NNOM_SRC/port $DST/
cp -rv $NNOM_SRC/src $DST/


echo "Weights file: $WEIGHTS"
mkdir -p $DST/$NET_NAME
cp -v $WEIGHTS $DST/$NET_NAME/

