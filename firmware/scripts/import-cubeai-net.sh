#!/bin/bash


if [ $# -ne 4 ]
then
  echo "Usage: "
  echo "  ./scripts/import-cubeai-net.sh <cube project dir> <net name> <destination for lib> <destination for src>"
  exit
fi

CUBE_SRC=$1
NET_NAME=$2
LIB_DST=$3
AI_SRC_DST=$4

echo "Copy CubeAI net from $CUBE_SRC"
echo "Library destination: $LIB_DST"
echo "Net destination: $AI_SRC_DST"

mkdir -p $LIB_DST
mkdir -p $AI_SRC_DST

# library
cp -rv $CUBE_SRC/Middlewares/ST/AI $LIB_DST/

# source files
cp -rv $CUBE_SRC/Src/aiPbMgr.c $AI_SRC_DST/
cp -rv $CUBE_SRC/Src/aiTestUtility.c $AI_SRC_DST/
cp -rv $CUBE_SRC/Src/aiValidation.c $AI_SRC_DST/
cp -rv $CUBE_SRC/Src/app_x-cube-ai.c $AI_SRC_DST/
cp -rv $CUBE_SRC/Src/${NET_NAME}.c $AI_SRC_DST/
cp -rv $CUBE_SRC/Src/${NET_NAME}_data.c $AI_SRC_DST/
cp -rv $CUBE_SRC/Src/pb_common.c $AI_SRC_DST/
cp -rv $CUBE_SRC/Src/pb_decode.c $AI_SRC_DST/
cp -rv $CUBE_SRC/Src/pb_encode.c $AI_SRC_DST/
cp -rv $CUBE_SRC/Src/stm32msg.pb.c $AI_SRC_DST/

# includes
cp -rv $CUBE_SRC/Inc/RTE_Components.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/aiPbMgr.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/aiTestUtility.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/aiValidation.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/app_x-cube-ai.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/bsp_ai.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/constants_ai.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/${NET_NAME}.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/${NET_NAME}_data.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/pb.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/pb_common.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/pb_decode.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/pb_encode.h $AI_SRC_DST/
cp -rv $CUBE_SRC/Inc/stm32msg.pb.h $AI_SRC_DST/

# patch the app file
patch $AI_SRC_DST/app_x-cube-ai.c -i scripts/app_x-cube-ai.c.patch
